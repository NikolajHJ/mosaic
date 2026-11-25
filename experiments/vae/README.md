# VAE training and generation

Question: improve prior generation and local robustness while retaining reconstruction.

The [training timeline](../../docs/vae-training-timeline.html) follows the VAE from ZINC through epoch 48, with separate entries for production changes and research pilots. The [Mosaic migration plan](../../docs/mosaic-retirement-plan.md) describes the proposed handoff after the current attention leg.

[Three-block encoder comparison](epoch48-encoder-upgrade.md) measures exact recovery, accessible edits and learning speed against the ordinary next leg of epoch 48.

[VAE research handoff for agents](../../docs/vae-research-handoff.md) traces the full model lineage, explains the loss variants and indexes the evidence and checkpoints.

The [training-data analysis](training_data_eda/experiment.md) measures size and ring topology across the full 60M training view, functional groups and scaffolds in a uniform chemical panel, and variance across molecular representations.

[Why Gaussian draws fill all 50 atom slots at epoch 47](occupancy-collapse.md)
traces the decoder, compares loss gradients, and tests recovery on held-out molecules and prior draws.

[Epoch-47 recovery continuation](epoch47-occupancy-continuation.md) switches
the large VAE's prior objective at the saved 47.68M-row checkpoint.
The [next allocation](epoch47-occupancy-continuation.md#5-dropout-in-the-next-a100-allocation)
adds 5% training dropout to its decoder atom updates at that boundary.
The [epoch-48 first leg](epoch47-occupancy-continuation.md#epoch-48-paired-first-leg) compares four and seven rounds with graph-matched reconstruction and 1:1 projected priors.
The [seven-round production arm](epoch47-occupancy-continuation.md#functional-group-loss-in-the-seven-round-arm) also adds weak aromatic-ring and amide yield targets of 8% and 12%.
The separate [weak functional-group pilot](weak-functional-groups.md) compares valid aromatic-ring and amide yields on matched random draws.
The [epoch-48 aromaticity investigation](training_data_eda/epoch48-aromatic-mechanism.md) isolates saturation of generated rings, dependence on coordinated latent values and the contribution of the teacher loss.
The [motif-completion comparison](training_data_eda/epoch48-motif-completion.md) tests a local bond-order loss on Gaussian priors and tracks ring sizes alongside raw generation and reconstruction.
The [shared aromatic-ring pilots](training_data_eda/shared-ring-pilots.md) compare selective teacher masking and a learned whole-ring decision from checkpoint 623116.
The [corpus-rate comparison](training_data_eda/epoch48-motif-completion.md#testing-the-corpus-percentages-in-the-trainer) tests aromatic and amide targets measured on the 60M training view and pools 240 prior draws per scheduled update.
The [seven-round ring-size comparison](epoch47-occupancy-continuation.md#ring-sizes-in-the-second-seven-round-allocation) tests whether weak penalties for excess three-membered and large rings improve generated graphs while preserving useful rings, scaffold diversity and reconstruction.

[435k graph VAE on V100](compact-benchmark.md) compares batch 256 and 512 with
training and evaluation that recognize atom-renumbered reconstructions.
The [first complete 60M training pass](compact-benchmark.md#first-complete-60m-training-pass)
uses batch 256, reconstruction and KL on real graphs, and the two-part atom-count correction on fresh random vectors.

[Bond-message decoder comparison](bond-message-comparison.md) tests whether learned soft bonds can replace triangle and attention updates while preserving cyclic structures and giving useful Protenix edit gradients.
The [longer three-decoder comparison](bond-message-comparison.md#longer-three-decoder-comparison) measures reconstruction and bond selection after 8,192 updates for each architecture.

[ChEMBL 37 audit and matched data comparison](chembl-comparison.md) measures
the additional molecules and scaffolds the current VAE can represent.
[Full combined-corpus continuation](combined-training.md) uses epoch 45,
canonical ordering and one fresh prior point per real molecule.

**Canonical-input pilot 29883264 completed on 3 November at 20:55 CET.** After
250 encoder-only and 750 joint updates, exact reconstruction matched epoch 45
at 2,044/2,048. On 10,000 shared samples, random validity was 75.48% argmax and
63.04% categorical, versus 75.55% and 63.05% for epoch 45. Exact reconstruction
with shuffled atom order reached 1,020/1,024. [Protocol and results](invariant-pilot.md).

**New separate branch, 3 November 2026:** [invariant encoder pilot](invariant-pilot.md)
adds invariant pooling and matching-aware reconstruction, initializes compatible
epoch-45 weights, then runs bounded encoder warm-up and joint fine-tuning.
Production epoch45 remains unchanged/paused; no interpolation loss is added.
Pilot **29878344 DONE at 14:47:51 CET**, both phases and comparison finished.
Order sensitivity falls to floating noise, but exact/valid reconstruction is
0/2,048; random argmax75.42% versus parent75.55%, categorical60.45% versus63.05%.
This branch is not promoted; epoch45 stays the usable baseline.

**Recovery pilot29879619 DONE3Nov16:52:55CET**,4,000+1,000 updates and evaluation
completed in23.28min. Topology-sensitive messages/teacher-only warm-up improve
code variance, but exact reconstruction remains0/2048. Warm-up chemically valid
outputs62.11%, final joint0.10%; random argmax69.33% versus parent75.55%, categorical
47.63% versus63.05%. Order-stable labels100% do not establish reconstruction.
Branch not promoted; see [saved outcome and diagnosis](invariant-pilot.md).
Production45 stays unchanged/paused; no interpolation or additional run started.

**Next frozen evaluation, 3 November 2026:** [matched epoch-45 interpolation](interpolation.md)
tests chemical-neighbor paths at 25%/50%/75%, with original-order and matched
re-encoded endpoint controls. Training remains paused; no interpolation loss
has been adopted into training. Job **29877725 DONE at 12:40:24 CET**, 128 pairs:
canonical midpoint validity 51.56%; identity-map pairs 30/30 valid. Matched
midpoint validity 98.44% is confounded by only 38/128 correct B endpoints.
Report and full denominator/ordering analysis are linked from the protocol.

**Current question, 3 November 2026:** [paired epochs 43–45 evaluation](checkpoint-comparison.md)
uses 10,000 fresh prior vectors, 10,000 held-out reconstruction molecules and
local perturbations. Job **29877255 DONE at 11:44:41 CET**, 14.39 minutes;
fresh prior argmax validity 75.49%/74.88%/75.41% for epochs 43/44/45, exact
reconstruction 99.84%/99.88%/99.87%. Training stays paused pending review and
the interpolation results; no training callback is attached.
The first launch failed before model loading; the replacement changes only
shell invocation, retaining the exact evaluation source/config.

Latest status, **2 November 2026 20:53 CET**: **epoch45 completed successfully**.
Native job29871778 is DONE at18:48:17 CET, launcher return0. All6,596 updates
finished, absolute296820. All18 final health checks, both terminal repair checks
and unchanged frozen anchors pass. Saved `epoch0045/epoch_0045.pt`, SHA
`5c4946c15515e70bd01281eea7b74693c18598c4b47f6724fcc9898deb1302be`.

Final fixed raw/strict/unique validity **769/1024 (75.10%)**, versus44's759/1024
(74.12%):+0.98 percentage points on this panel. Exact/valid reconstruction
256/256; expected sulfur1.525%. Training35,898.02s (**9.97h**), versus44's9.94h;
the earlier sparse timing advantage did not become an end-to-end speedup.

The final VAE callback29871779 and Protenix pilot29873719 are PEND because
their CPU/GPU queues are **inactivated by the administrator**, not because the
VAE prerequisite failed. No Protenix execution yet; no jobs were moved or
resubmitted. [Final metrics](../../outputs/hpc-results/source157-vae-protenix-latent-attempt001/status-1793649179553656800.json),
[exact terminal history and inactive-queue reasons](../../outputs/hpc-results/source157-vae-protenix-latent-attempt001/terminal-check-1793649216712916700.json).

Latest status, **2 November 2026 12:34 CET**: epoch45 attempt002 job **29871778
RUN** on `n-62-18-9`, exclusive GPU mode; endpoint callback29871779 waits normally.
Its32-update recovery probe and callback passed earlier; full training restored
saved44 and started08:48:20 CET with the unchanged structural proposal policy.

Latest panel at **2,500/6,596 updates (37.9%)** passes all18 checks. Fixed raw/
strict and unique validity **765/1024 (74.71%)**, versus44 final759/1024 (74.12%);
exact/valid reconstruction256/256. Previous panels at2,300/2,400 were766/756,
so a sustained validity improvement is still not established. Expected sulfur
1.352%. No failure record; occasional native approximate-ring warnings are
retained in stderr. Mean of25 logged update samples5.248s; current progress
suggests completion around18:30--19:00 CET, subject to runtime/evaluation changes.
No jobs/settings changed in this status check.
[Latest progress, timing and scheduler evidence](../../outputs/hpc-results/source156-vae-epoch45-structural-attempt002/status-1793619267942003000.json).

Latest recovery, **2 November 2026 08:24 CET**: the administrator killed epoch-45
job 29866013 on **1 November at 23:52:27**, terminal EXIT130/TERM_ADMIN at23:52:36.
The actual scheduler history confirms this VAE already requested
`num=1:mode=exclusive_process:j_exclusive=yes`; it was still included in the
account-wide termination described in the user's email. Its last health panel
at2,900 updates passed all18 checks:754/1024 (73.63%) raw/unique validity,
256/256 exact/valid reconstruction. No completed45 checkpoint was saved; the
partial diagnostic is not promoted. Saved epoch44 was SHA-checked again and is
the recovery input. [Native kill history, logs and health](../../outputs/hpc-results/source156-vae-epoch45-structural-attempt001/exclusive-email-status-1793604041899688700.json).

Restart **source156 structural attempt002** reuses the exact existing source
archive and all scientific settings, including structural proposals, sampling,
loss, optimizer/RNG, anchors, health checks and endpoint45. Only campaign output,
home-control path and job-name prefix change. Old outputs/control stay intact;
no other repository's jobs were changed. Duplicate-writer checks found no live
VAE, and terminal44/source/checkpoint evidence passed before submission.

At08:23 CET, new32-update GPU probe **29871688 PEND** and dependent callback
**29871689 PEND** were submitted. LSF confirms the exclusive GPU request. A
passing probe automatically submits full45 from the original44 checkpoint;
probe updates are discarded. Probe4h/full45 24h allocations,16 CPU slots,
4GB per slot,80GB A100; existing internal deadlines/checks unchanged. Training
has not restarted yet. New study SHA
`ce78967853aa5b14e7db4c19f4e967156a777287501d1e30da7ad1929c27f1e5`.
[Actual recovery study](../../outputs/hpc-submissions/source156-vae-epoch45-structural-attempt002.json),
[submission/native exclusive request](../../outputs/hpc-submissions/source156-a002-submit-1793604225437398800.json).

At08:25 CET the recovery probe remains PEND: the queue cannot yet supply the
16 CPU affinity slots alongside the GPU;12 slots are reserved. LSF currently
estimates no earlier than **3 November 07:49:43 CET**, subject to scheduling
changes. The callback has the correct ended(probe) dependency. This is capacity
waiting, with no numerical check executed in attempt002 yet.
[Latest native request, pending reason and dependency](../../outputs/hpc-results/source156-vae-epoch45-structural-attempt002/status-1793604299038814700.json).

Comparison, **1 November 2026 21:28 CET**: epoch 45 remains RUN, latest complete
health panel at 1,200/6,596 updates (18.2%). Fixed 1,024-latent raw/strict validity
is **733/1024 (71.58%)**, versus epoch 44's final **759/1024 (74.12%)**: -2.54pp.
Earlier epoch-45 panels reached 782/1024 (76.37%); there is no sustained validity
gain yet. Exact/valid reconstruction remains 256/256, all 18 checks pass, no
representation failures, expected sulfur 1.455% versus 1.400% at epoch 44.
Health thresholds retain the original campaign baselines; passing is not proof
of improvement over the immediately preceding checkpoint.

For 12 matching logged update indices (100,200,...,1200), mean update time is
**5.472s in epoch 44 vs 4.815s in epoch 45 (12.0% less time)**. Reported teacher/
search time is 3.546s vs 2.822s (20.4% less); prepass 1.138s vs 1.182s. Search
work overlaps model work, so component times are not additive. These are sparse
logged timings on separate A100 nodes and different epoch batches; they support
an observed speed improvement, not isolated causal attribution or a final epoch
runtime. Epoch 44's completed wall time was 9.94h.
[Saved comparison](../../outputs/hpc-results/source156-vae-epoch45-structural-attempt001/epoch44-45-comparison-1793564873727023700.json),
[all epoch-45 panels](../../outputs/hpc-results/source156-vae-epoch45-structural-attempt001/validity-trend-1793564905900084100.json).

Latest status, **1 November 2026 21:26 CET**: **epoch 45 job 29866013 RUN** on
`n-62-18-12`. Its actual handoff confirms the saved epoch-44 checkpoint and
`fixed-atom-proposals-v1` with component joining/dense pruning and sampling off.
Latest health panel at absolute update 291424 (1,200 of 6,596 epoch updates)
passes all 18 checks. Endpoint callback 29866014 waits normally. Epoch 45 is
not complete; no settings changed during this read-only confirmation.
[Live handoff and health receipt](../../outputs/hpc-results/source156-vae-epoch45-structural-attempt001/confirmation-1793564791141773800.json).

Latest status, **1 November 2026 19:18 CET**: the structural-proposal GPU probe
**passed all 32 updates**, all 18 final early-health checks and both terminal
repair checks; frozen anchors are unchanged. Probe-panel strict/raw validity
765/1024 (74.71%); exact and valid reconstruction 256/256. Measured mean update
5.153s; conservative full-epoch projection 12.15h including the existing margin.
These are bounded-probe results, not an epoch-45 outcome.

The corrected callback succeeded and automatically submitted **epoch 45 job
29866013 at 12:13 CET**, plus endpoint callback 29866014. The training job is
still **PEND**, with 16 CPU slots reserved and the requested physical GPU not
yet available. `gpua100` is Open:Active. Full training has not started; no job
or settings were changed during this status check.
[Compact probe and scheduler evidence](../../outputs/hpc-results/source156-vae-epoch45-structural-attempt001/probe-passed-epoch45-pending-1793557117710871800.json).

Latest status, **1 November 2026 12:12 CET**: GPU probe **29865924 RUN** on
A100 node `n-62-18-8`. Epoch-44 checkpoint, optimizer/RNG and the selected
structural policy restored successfully. The first **10 updates passed all 18
early-health checks**. The 32-update final result is not written yet; callback
29865925 waits normally, and full epoch 45 has not started.
[Native handoff and first health result](../../outputs/hpc-results/source156-vae-epoch45-structural-attempt001/progress-1793531519891019000.json).

Latest adoption, **1 November 2026 12:06 CET**: the user authorized recovering
the scheduler handoff and adopting structural proposals for epoch 45. New
source156 `vae-epoch45-structural-attempt001` is uploaded; **32-update GPU probe
29865924 PEND**, callback **29865925 PEND** on its normal dependency. The probe
has an A100 reservation and is waiting for host job slots. A native pass is not
yet established, and epoch 45 is not yet submitted. The callback queues it
automatically only after the probe and exact predecessor checks pass.

The corrected status guard accepts exact job-specific terminal `bhist` DONE
evidence while `bjobs` is stale or the record has aged out, retains raw evidence,
rejects EXIT and older DONE followed by restart, and retries briefly for missing
proof. The old source155 controller and failed callback receipts stay unchanged.
The new controller has already verified epoch 44's terminal history. It will
also use the fix after epoch 45.

Epoch 45 uses the selected deterministic structural proposal builder: component
joining, connectivity-preserving dense pruning and existing beam fallback;
sampling is off. Target discovery changes, while the existing full-graph repair
NLL normalization/weights, real reconstruction/KL, composition, epoch-35 occupancy
and epoch-36 valid-reference anchors, health checks, 256 real/512 synthetic counts
and 0.125 AdamW displacement remain unchanged. The existing 25% local synthetic
mixture and its RNG/ramp are restored from epoch 44 (original ramp start 283628).
The probe updates are discarded; full epoch 45 starts from the exact checkpoint.
Endpoint remains 45, with a 24h allocation/23h internal budget and a checkpoint plus
generation diagnostics. No other job or source checkpoint was changed.

The 47,687,540-byte epoch-44 input was SHA-checked remotely and in a local backup.
All 44 relevant local cases passed and passed again against the exact packaged
snapshot; both actual CLI help commands and real checkpoint lineage metadata passed.
659 predecessor archive members are unchanged; the existing regularization
implementation body is identical after adding its grouped import path.
Archive SHA `fafb64d34a206bfce7b55d61f9e8f00fb6d96111034e0cc6e87e216df4a9461e`,
study SHA `80e9024ae1b5ddb90eace0c697e5152fffc832f078d2e97b6c5168d9fb446108`.
[Actual study](../../outputs/hpc-submissions/source156-vae-epoch45-structural-attempt001.json),
[submission and real terminal proof](../../outputs/hpc-submissions/source156-submit-1793531118499631900.json),
[queue receipt](../../outputs/hpc-results/source156-vae-epoch45-structural-attempt001/status-1793531165908655000.json).

Previous status, **1 November 2026 11:53 CET**: epoch44 completed and saved
`epoch0044/epoch_0044.pt`, SHA
`637745ad896b02fa9016d23f97a8826597373dc1f81953dc3ebaafb9c94d0683`,
at290224 absolute updates (6,596 updates this epoch). Measured training time
35,781.36s (**9.94h**), mean5.374s/update. All18 final early-health checks and
both terminal search/coverage checks pass; frozen anchors are unchanged.
Fixed Gaussian raw/unique validity **759/1024 (74.12%)**, versus765/1024 (74.71%)
at43. Exact/valid reconstruction **256/256**, expected sulfur **1.3995%**.
This small validity decline does not establish improvement or collapse.

GPU29861136 is confirmed **DONE at04:41:35CET** by exact job-specific history.
Callback29861137 ran04:41:36--04:41:39 and **EXIT1** at `require_done(29861136)`:
`Predecessor allocation did not finish DONE`. Epoch45 has no receipt/output/job
and is **not queued or running**. The immediate callback failure is consistent
with scheduler propagation delay; its exact transient `bjobs` response was not
saved, so that mechanism is not established. Training itself and the saved44
checkpoint succeeded; only the scheduling handoff needs recovery. This status
check changed no jobs, control sources, losses or checkpoints.
[Saved metrics/current queues](../../outputs/hpc-results/source155-vae-local-neighborhood-attempt002/progress-1793530395521016200.json),
[job histories and callback traceback](../../outputs/hpc-results/source155-vae-local-neighborhood-attempt002/epoch44-handoff-1793530418911913600.json).

Previous status, **31 October 2026 21:50 CET**: scheduling is active again. Allocated
launcher check29860450 passed, GPU probe29860453 passed all100 updates, and
callback29860454 completed the exclusive handoff. Epoch44 job **29861136 RUN**
on an80GB A100 since18:43CET, resumed from the authentic43 checkpoint (not the
disposable probe state). Its latest saved panel is at2,000/6,596 updates (~30%):
raw Gaussian argmax validity **769/1024 (75.10%)**, versus765/1024 (74.71%) at43;
exact/valid reconstruction **256/256**, expected sulfur **1.4243%** versus1.4734%.
All18 early-health checks pass. This small panel change is not evidence of a
reliable improvement;44 has no completed checkpoint yet. Callback29861137 is
pending normally behind44 and will advance to45 only if the inherited checks
pass. The old held callback29849761 was retired by the successful handoff.

The100-update probe took677.56s, kept frozen anchors unchanged and passed both
terminal search/coverage checks. Probe validity was754/1024 (73.63%); passing
means bounded startup/update/health evidence, not an improvement or derivative
parity claim. Latest full-epoch updates take about5.2s each, excluding panel
overhead. No loss, sampling, resource or safeguard settings changed during this
read-only status check. [Native results and scheduler history](../../outputs/hpc-results/source155-vae-local-neighborhood-attempt002/status-1793479814380286600.json),
[current queue/progress receipt](../../outputs/hpc-results/source155-vae-local-neighborhood-attempt002/progress-1793479838906214900.json).

Earlier recovery, **31 October2026 13:41CET**: the epoch43 checkpoint is readable,
SHA-verified and backed up locally. Its fixed Gaussian validity is765/1024
(74.71%), versus42's782/1024; exact/valid reconstruction remains256/256.
The earlier source155 probe failed before training on a BeeGFS scratch read;
its callback also stalled before it could recover the handoff. The underlying
storage incident is not independently established; current targets report
Online/Good and affected files now read successfully.

The same campaign runner now supports home-storage control/logs, system Python,
bounded scratch reads and checksum-verified node-local source staging. Startup
must reach the authenticated model/input handoff within600seconds. Exact
`bhist` DONE proof handles the aged-out source43 job; new training cannot wait
on an impossible aged-out scheduler dependency. The loss,1:2 count,0.125
displacement, local sampling policy, anchors and through45 endpoint are unchanged.

Source155 attempt002: allocated launcher check **29860450 PEND**,100-update
GPU probe **29860453 PEND** behind that check, callback **29860454 PEND**.
**At that check all cluster queues showed `Open:Inact_Adm`; dispatch was paused.**
After the launcher check and GPU probe pass,44
resumes the exact completed43 checkpoint and45 follows serially. The original
callback29849761 remains held for exclusive handoff; a failed probe preserves43
without automatically running the old settings. No new44 training has started.

Validation at13:41:23 focused local cases passed;22 passed again against the exact
packaged snapshot. Allocated launcher/native GPU results were still pending then.
[Recovery and raw evidence](../../outputs/hpc-results/source155-vae-local-neighborhood-attempt002/recovery-summary-20261031.json).

Command/config: the study snapshot is
[`source155-vae-local-neighborhood-attempt002.json`](../../outputs/hpc-submissions/source155-vae-local-neighborhood-attempt002.json).
On HPC, the home-storage `vae_neighborhood_campaign.py --study STUDY --study-sha256 SHA`
uses the existing `probe`/`after-probe`/`run-epoch`/`advance` modes. The100-update
probe has4h; each full epoch has24h with23h internal budget. The new
[`hpc_launch.py`](../../scripts/hpc/hpc_launch.py) is shared launcher code;
[`check_launcher_allocated.py`](check_launcher_allocated.py) tests staging and a
deliberate startup stall without importing a model. Do not replay submission
commands while their saved receipts/jobs exist.

- Reusable entry point: `python -m latentmol.training.workflow --help`.
- Shared code: [training](../../src/latentmol/training/README.md), [models](../../src/latentmol/models/).
- Baseline configs: [AE smoke](../../configs/ae_smoke.yaml), [large triangle](../../configs/vae_large_triangle.yaml).
- Saved continuation: [epochs40–45](../../docs/80_epoch40_to45_continuation.md).
- New local-neighborhood proposal/actual dated progress: [study89](../../docs/89_latent_neighborhood_continuation.md).

Next: use the saved study record for the actual checkpoint/handoff. This index
makes no new submission or adoption. Related guidance comparisons belong in
[chemistry guidance](../chemistry-guidance/README.md).

## Are invalid prior samples outside the learned molecule distribution?

Investigated **31 October 2026 23:45 CET** using the saved epoch43 checkpoint
(`0f5d4acc2520769de53fe788ef966a35b0b052490adb78dee90097436da53e08`).
Question: is the ~75% raw prior validity an intrinsic VAE limit, a Gaussian-tail
problem, or a gap between reconstruction and generation? No training, repair,
HPC submissions, checkpoint writes or changes to the running44/45 chain.

Bounded local CPU evaluation: four threads, two seeds904017/140041,600s cap per
seed, actual54.64s/57.27s. Reference mixture uses1024 byte-verified training
graphs; evaluation uses256 disjoint validation molecules. Reconstruction at
the mean is deterministic and repeated on the same256 molecules; local and
posterior panels have two draws per molecule, not512 independent source
molecules. The prior panel has1024 random latent points across the two seeds.

| Latent source | Valid raw argmax outputs |
| --- | --- |
| Real validation molecule, encoder mean |256/256 (100%); exact labels256/256 |
| Mean plus local0.05 posterior-standardized RMS perturbation |512/512 (100%) |
| Full encoder posterior sample |499/512 (97.46%) |
| Independent standard Gaussian prior |763/1024 (74.51%) |
| Same prior directions scaled by0.75 |747/1024 (72.95%) |
| Same prior directions scaled by1.25 |791/1024 (77.25%) |

The distinction is reconstruction versus generation: random prior points have
no original molecule to reconstruct. Real posterior samples and prior samples
both have typical Euclidean radius~16, but very different validity. Within the
prior panel, valid/invalid mean radii are15.988/15.981; radius ranking AUC0.501
does not identify failures. Shrinking random vectors therefore is not supported
as a generic fix. The1.25-scale change varied strongly by seed (406vs377 valid
in the first,385vs386 in the second); no alternate sampling scale is adopted.

Failures are138 disconnected graphs and123 sanitization failures, with zero
input/representation failures. Of261 invalid argmax outputs, **177 (67.8%)**
yielded at least one valid categorical graph within four draws from the same
decoded probabilities. Across all4096 categorical draws,2613 (63.79%) are valid;
best-of-four selection costs extra sampling and does not make individual draws
more valid. All512 tiny local perturbations decode the original molecule;486/512
full posterior draws also do so. High local validity is therefore not evidence
of new molecular discovery.

An equally weighted1024-component mixture of real training posteriors gives a
finite-reference `log q(z)-log p(z)` diagnostic. It does **not** reliably separate
valid/invalid prior points (pooled AUC0.475, active-dimension version0.475).
This sparse high-dimensional estimate is not a certified OOD classifier: neither
its failure nor the good posterior validity proves or disproves a full-data
aggregate-posterior mismatch. There are59 active mean dimensions by the
variance>0.01 convention, not evidence that every other coordinate is useless.
Density math matched `torch.distributions`; mixture permutation and tied-score
ranking fixtures passed. Model state was unchanged in both evaluations.

Interpretation: reconstruction is at this small panel's measurement ceiling;
there is no evidence of a universal~75% VAE validity ceiling. Learned posterior
regions decode more reliably, but the tested failures cannot simply be labeled
extreme Gaussian outliers. Fresh prior points already receive chemistry training,
so they are not outside the synthetic training distribution. Imperfect global
graph consistency and differences between the data posterior and the sampling
prior remain plausible contributors. VAE prior/posterior mismatch is established
in the literature ([Dai and Wipf](https://arxiv.org/abs/1903.05789),
[Saha et al.](https://arxiv.org/abs/2311.07693)); it is not proven as the cause of
each failed point here. Finish the currently authorized44/45 comparison, then
compare paired independent generation panels before choosing another training
change. A larger model, stronger KL or a density filter is not justified by
these diagnostics alone.

Reproduce with the existing downloaded inputs (one new output per seed):

```sh
python experiments/vae/diagnose_prior_support.py --checkpoint EPOCH43.pt --inputs INPUTS.json --output NEW_OUTPUT --prior-count 512 --seed 904017 --seconds 600
```

Repeat with seed140041. The script consumes the bounded cohort exported by
`scripts/local_neighbor_inputs.py`; it does not require the full HPC cache.
[Combined result](../../outputs/runs/vae-prior-support/epoch43-summary.json),
[seed904017](../../outputs/runs/vae-prior-support/epoch43-001/result.json),
[seed140041](../../outputs/runs/vae-prior-support/epoch43-002/result.json).
Each output retains every raw valid/invalid graph in `attempts.jsonl`.
