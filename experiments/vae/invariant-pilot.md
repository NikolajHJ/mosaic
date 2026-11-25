# Short invariant-encoder and matching-aware reconstruction pilot

## Lossless canonical-input correction — 3 November 2026

**Job 29883264 finished on 3 November at 20:55:53 CET.** The V100 allocation took
26m30s and completed all 250 encoder-only and 750 joint updates plus evaluation.

| Shared evaluation metric | Epoch 45 | After joint training |
| --- | ---: | ---: |
| Exact reconstruction, 2,048 validation molecules | 99.80% (2,044) | 99.80% (2,044) |
| Chemically valid reconstructed outputs | 99.95% (2,047) | 99.95% (2,047) |
| Random argmax validity, 10,000 samples | 75.55% | 75.48% |
| Random categorical validity, 10,000 samples | 63.05% | 63.04% |
| Exact reconstruction after atom-order shuffling, 1,024 trials | 0.00% | 99.61% |

The same 2,044 molecules reconstructed exactly in both models. The paired
validity differences were −0.07 percentage points for argmax (approximate 95%
interval [−0.23, +0.09]) and −0.01 for categorical sampling ([−0.16, +0.14]).
After joint training, all 1,024 permutations reproduced the corresponding
unshuffled outputs and posterior parameters exactly. On the fixed 128-molecule
monitoring panel, final teacher-code MSE was 5.04e-6 and the variance ratio was
0.99935. The run verified the frozen parent checkpoint hash.
[Final comparison](../../outputs/hpc-results/vae-canonical-invariant-pilot-20261103-a01/completed-comparison.json),
[completion record](../../outputs/hpc-results/vae-canonical-invariant-pilot-20261103-a01/outcome-1793780087350619600.json).

Job 29883264 started at 20:29:23 CET on V100 host `n-62-11-13`.
Four CPU/16GiB host memory, one exclusive32GB V100 (30GiB reservation),4h limit.
Worker ready after97.63s. **Native mapping check passed on64 shuffled training
graphs**, full labels identical and max mean/logvar error0. Initial128 validation
molecules have100% exact/valid reconstruction, teacher MSE0 and variance ratio1.
At the 20:46 CET check, warm-up had completed and joint training reached 540/750.
At the joint 500 health panel, exact and valid reconstruction
remains128/128, and prior argmax validity is389/512=75.98% versus390/512=76.17%
at the identical epoch45 start. The teacher-code variance ratio was 1.00090 and
MSE was 5.71e-6. The logged gradients were finite and unclipped.
[Interim progress and health](../../outputs/hpc-results/vae-canonical-invariant-pilot-20261103-a01/status-1793735159923211800.json).
LSF recorded a GPU power/clocks event at20:31:52 and explicitly says the GPU can
still perform workload. The job subsequently completed at about 1.19s per joint
training update.
Source SHA256 `428ec15125674f2f7298dd5bc71f3d4bb1bbfa05f50717cca3c1707d2c86a684`.
Run `/work3/s234818/latentmol/runs/vae-canonical-invariant-pilot-20261103-a01`.
[Build/resources](../../outputs/hpc-submissions/vae-canonical-invariant-pilot-20261103-a01/build.json),
[reconciled accepted submission and native RUN](../../outputs/hpc-submissions/vae-canonical-invariant-pilot-20261103-a01/submission-reconciled.json),
[local/exact-source validation](../../outputs/hpc-submissions/vae-canonical-invariant-pilot-20261103-a01/local-validation.json).
[Native GPU startup/mapping/initial panel](../../outputs/hpc-results/vae-canonical-invariant-pilot-20261103-a01/startup-1793734307440783000.json).
Protected parent/manifest/GPU-helper hashes matched; quota89.33/100GiB, own
expected bounded outputs below1GiB. The one-off operator missed the accepted ID;
the native receipt/scheduler were reconciled without repeating `bsub`.

Question: can we remove input atom-order dependence **without discarding the
molecule-specific information learned by epoch45**? Config:
[invariant-pilot-v3.json](invariant-pilot-v3.json), same runner/command below.
This is an explicit change of approach: **a discrete canonical-order front end
plus the retained ordered neural encoder**, not another claim that a newly
trained neural set encoder has learned the parent's molecular representation.

### Why the learned set encoder still fails

The saved v2 curves show training teacher MSE~0.16–0.17 from update100 through
4,000, close to validation0.16382 and constant-mean baseline0.17416. Logged
gradients after100 are finite, nonzero and below the clipping threshold. This
does not support an explanation based only on held-out overfitting or clipping.
The network learns broad information while failing to recover individual codes.

Removing positions and replacing the full50×96 readout with193 pooled moments
also changed what can be represented. Four rounds of local message passing and
sum/mean/count pooling do not preserve all global connectivity. Two distinct
sanitized carbon cage graphs (each six degree-three carbons, one with two
triangles and one with none) produce **exactly identical codes even after the
sum/degree fix**. This holds structurally for identical-node regular graphs:
every node sees the same local feature multiset at every round. A richer pooled
MLP alone cannot recover information already erased. These cages are a proof
of an architectural collision, not a claim about its frequency in MOSES or that
it alone accounts for every failed example. Regressing the ordered teacher's
identity-specific coordinates from the compressed view can favor mean-like codes.
[CPU counterexample](../../outputs/hpc-results/vae-canonical-invariant-pilot-20261103-a01/regular-cage-counterexample.json),
[saved training-curve analysis](../../outputs/hpc-results/vae-invariant-matched-recovery-20261103-a01/training-analysis-1793733562630586000.json).

### Correction and boundaries

- `ModelConfig.encoder_pooling="canonical"` keeps the original positions,
  bond-aware message layers, flattened50×96 readout, pooled MLP and256-dimensional
  Gaussian heads. Every encoder and decoder tensor is copied exactly. Total
  parameters3,941,860, same as epoch45; neural set v2 had2,463,204.
- [Canonical slot mapping](../../src/latentmol/chemistry.py) uses the same
  canonical stereo-free SMILES convention as dataset preparation. Obtain a full
  atom bijection, append padding slots and verify **every** atom, formal charge
  and both bond axes exactly match the canonical encoded labels. A mismatch
  rejects the input. No chemistry repair, neutralization, aromatization, fragment
  selection or sampling is allowed to silently change labels.
- [Encoder input adapter](../../src/latentmol/datasets/ordering.py) requires
  finite, exactly one-hot hard-graph inputs with no required input gradient.
  Soft/invalid/unrepresentable inputs fail explicitly. Encoder parameter
  gradients remain live. The latent→decoder→relaxed-scorer gradient path never
  enters this preprocessing and is unchanged. RDKit runs outside that path.
- For supported hard inputs, canonicalization handles atom-order invariance
  algorithmically while the old neural encoder retains molecule identity.
  It does **not** establish a smooth neural encoder on relaxed graphs, globally
  smooth latent neighborhoods, or interpolation behavior. No interpolation loss.

### Short pilot

Same frozen45 input/hash and65,536-molecule train subset, batch64, seed11034601.
Initialize all parent tensors identically; only this separate branch is trained.
An allocated startup check on64 training graphs permutes all slots, restores
identical full labels and compares posterior means/logvars with the parent.
This is a mapping/function check, not a reconstruction quality admission gate.

1. **250 encoder-only updates** with deterministic matching-aware reconstruction,
   normalized teacher-mean alignment weight1 and logvar MSE weight0.1; decoder
   frozen. Encoder LR1e-5. No need to relearn discarded positions or readout.
2. **750 joint updates**, stochastic matching-aware VAE loss, beta0.001,
   teacher-mean alignment2/logvar0.1 and the existing0.25 prior-decoder KL on64
   random codes. Encoder LR1e-5, decoder LR1e-6. Both rates are10× below v2
   joint rates because this starts from a functioning encoder/decoder pair.

Train-only calibration1,024 molecules; new optimizers, weight decay1e-5, clip5.
Total64,000 real graph views rather than v2's320,000, so this tests preservation
and short adaptation, not equal-budget training of a new neural architecture.
No expensive chemistry objective or parent real/random chemistry ratio is used;
random codes receive only the stated frozen-decoder anchor. Health/snapshots
every250, same2,048 validation reconstructions/10,000 prior points/256×4
permutations and perturbation panels. Quality changes are recorded through the
bounded endpoint; nonfinite/detached required paths, incompatible input/mapping,
frozen writes and collisions stop the affected operation. No callback/promotion.

**69 local CPU checks passed**, including exact label restoration for aromatic,
charged, sulfur, phosphorus and cage examples; retained teacher codes/parameter
gradients; explicit rejection of soft/input-gradient/normalizing conversions;
legacy strict loading; and complete tiny v1/v2/v3 training/evaluation pipelines.
All69 passed against the exact staged source with imported path asserted;
launcher shell syntax passed. Native GPU startup evidence is recorded above;
this is not yet the final2,048-molecule reconstruction/10k-prior comparison.
Epoch45 remains unchanged/paused.

## Recovery pilot — 3 November 2026

**29879619 DONE at16:52:55CET.** All4,000 teacher-only and1,000 joint updates,
followed by the shared evaluation, completed in1396.57s (23.28min scientific
runtime). Finite/attached paths and unchanged frozen parent checks passed.
**Reconstruction did not recover; do not promote this branch.**

| Metric | Original45 | Recovery warm-up | Recovery joint |
| --- | ---: | ---: | ---: |
| Exact identity reconstruction,2,048 validation molecules | 99.80% (2044) | 0% (0) | 0% (0) |
| Chemically valid reconstructed outputs | 99.95% (2047) | 62.11% (1272) | 0.10% (2) |
| Random argmax validity,10,000 shared latents | 75.55% | 75.55% | 69.33% |
| Random categorical validity,10,000 draws | 63.05% | 63.05% | 47.63% |
| Identical output labels under1,024 input permutations | 0% | 100% | 100% |

Validity here is connected, nonempty, radical-free sanitized/reparsable chemistry;
representation compatibility is also reported separately in the raw comparison.
Strict compatible warm-up reconstruction validity is1270/2048=62.01%; two
chemically valid outputs fail representation checks. Final2046/2048 outputs fail
chemical sanitization. Warm-up prior preservation is guaranteed by its frozen
decoder; it does not demonstrate encoder reconstruction success.

The revised encoder remains far from learning distinct teacher codes. On128
fixed validation molecules, warm-up mean-code variance is0.01137, only6.53% of
the teacher's0.17416; effective rank6.27. Teacher MSE0.16382 is only modestly
better than the constant-mean predictor0.17416. After joint updates these are
0.01121/rank6.30/MSE0.16474. This improves the old rank~1.6 collapse but still
does not recover the teacher's molecular information. The structural collision
fix remains valid; it was insufficient to make this warm-start protocol work.
Joint reconstruction loss decreases while molecular identity stays incorrect
and validity deteriorates, showing that the monitored training surrogate alone
is inadequate evidence of reconstruction success. This pilot does not isolate
matching approximation, coordinate-transfer difficulty and readout capacity.

Output-label order invariance is100%, but posterior rtol1e-5/atol1e-6 passes
1022/1024 warm-up trials and1013/1024 joint trials; roundoff exceptions remain
recorded. Prior argmax drops6.22pp (paired approximate95% interval[-7.30,-5.14]pp),
categorical drops15.42pp ([-16.59,-14.25]pp). These pointwise intervals are
unadjusted for multiple comparisons. No new training or submission followed
collection; epoch45 stays unchanged and production training paused.

[Readable native summary](../../outputs/hpc-results/vae-invariant-matched-recovery-20261103-a01/completed-summary.md),
[full shared comparison](../../outputs/hpc-results/vae-invariant-matched-recovery-20261103-a01/completed-comparison.json),
[compact outcome and latent diagnostics](../../outputs/hpc-results/vae-invariant-matched-recovery-20261103-a01/result-digest.json),
[native DONE/log/health receipt](../../outputs/hpc-results/vae-invariant-matched-recovery-20261103-a01/outcome-1793733225249734200.json),
[verified downloads](../../outputs/hpc-results/vae-invariant-matched-recovery-20261103-a01/completed-downloads.json).

Question: does preserving neighbor multiplicity and learning teacher coordinates
before reconstruction recover a useful invariant encoder? New config:
[invariant-pilot-v2.json](invariant-pilot-v2.json), same runner/command below.
Epoch45 and the failed first pilot remain unchanged; no interpolation loss.

Job29879619 ran on gpuv100 from3Nov16:29:30 to16:52:55CET on `n-62-20-14`, one
second after submission. Model/input-ready marker verified after99.38s. One exclusive
32GB V100 requested (30GiB reservation), four CPU/16GiB host memory,4h maximum;
no continuation callback. Completed results are above.
Source archive SHA256
`4d2cccfbe9e182e3dfde602f9d236b273e776e480e915b2f540fab00f94668d3`.
Run: `/work3/s234818/latentmol/runs/vae-invariant-matched-recovery-20261103-a01`.
[Build/resources](../../outputs/hpc-submissions/vae-invariant-matched-recovery-20261103-a01/build.json),
[native submission](../../outputs/hpc-submissions/vae-invariant-matched-recovery-20261103-a01/submission.json),
[local validation](../../outputs/hpc-submissions/vae-invariant-matched-recovery-20261103-a01/local-validation.json).
Epoch45, dataset manifest and GPU routing helper hashes matched preflight.
Quota86.95/100GiB leaves13.05GiB for this bounded, approximately1GiB-output pilot.

### Mechanism found in the existing evidence

The failed encoder collapsed, not merely reordered its output. Across the same
128 saved validation anchors, teacher mean codes had effective rank34.55 and
mean between-molecule variance0.17497. Warm-up reached rank1.64/variance0.00105;
joint reached rank1.83/variance0.00900. Warm-up mean-code norm was13.84 and norm
of the mean vector13.83: almost one common biased code. Teacher MSE0.922 was
even worse than predicting the teacher's constant mean (MSE0.175).
[Saved-data analysis](../../outputs/hpc-results/vae-invariant-matched-pilot-20261103-a01/saved-latent-collapse-analysis.json).

There is a specific structural information loss in the former invariant encoder:
degree-normalized **mean** messages cannot distinguish identical neutral carbon
neighbors at different degrees. All four carbons in butane, isobutane and
cyclobutane therefore get the same state at every layer, giving identical pooled
codes for these three different graphs. A local counterexample measured maximum
L2 difference4.17e-8 (roundoff). Fixed-scale sum messages distinguish them;
explicit typed-degree features reinforce that information.
[Counterexample](../../outputs/hpc-results/vae-invariant-matched-pilot-20261103-a01/carbon-topology-counterexample.json).
This is a proven architectural collision, not proof it alone explains every
failed molecule. Ordinary message passing still cannot distinguish every possible
nonisomorphic graph; the change removes this concrete failure, not that general limit.

The averaged readout also did not preserve the ordered teacher's function.
Immediately applying reconstruction through a frozen decoder offered a shortcut
to a low-loss common invalid template before latents were aligned; joint training
then removed latent alignment entirely. The saved collapse and biased means
support this training explanation, but do not isolate the contribution of every
loss term in a factorial experiment.

### Targeted correction and budget

- Opt-in encoder sums bond-specific neighbor messages divided by a **fixed3**,
  retaining degree information, and projects the four typed neighbor counts
  into node states (384 additional encoder parameters, initialized std0.02).
  Occupancy-masked invariant sum/mean/count pooling remains. No position features.
  Historical configs still default to degree means and have no new tensor keys.
- **4,000 teacher-only encoder updates**, AdamW LR3e-4. Train deterministic means
  and log-variances against the frozen canonical epoch45 encoder; **no decoder
  forward, graph matching or reconstruction gradient in warm-up**. Normalize mean
  MSE by the teacher's average between-molecule variance, calibrated on1,024
  training molecules only. Mean weight1, log-variance MSE weight0.1. This prevents
  the reconstruction branch from competing with the initial coordinate alignment.
- **1,000 joint updates**, matching-aware stochastic VAE reconstruction plus
  beta0.001 KL and the existing0.25 frozen-decoder prior KL on64 random codes.
  Retain normalized mean alignment weight2 and log-variance MSE weight0.1.
  Encoder LR1e-4, decoder LR1e-5, weight decay1e-5, clip norm5 unchanged.
- Same65,536 real training subset, batch64 and train/permutation seeds. This is
  320,000 graph views versus128,000 before, with a cheaper encoder-only warm-up;
  it is a recovery experiment, not an equal-training-budget architecture comparison.
- Fixed validation/prior health every250 updates; immutable snapshots every1,000
  and each phase endpoint. Add teacher MSE, normalized error, code bias norm,
  between-molecule variance, teacher variance ratio and effective rank to health
  reports. These expose collapse even when order invariance looks perfect.
- Same final panels as the first pilot:2,048 held-out reconstructions,10,000
  shared Gaussian latents with argmax/categorical decoding, and256 molecules ×4
  permutations. No production promotion/callback or interpolation is automatic.
  Quality findings do not abort this bounded pilot; finite/attached paths,
  input compatibility, frozen states and output collisions remain checked.

Local validation: **33 focused CPU cases passed**, then33 against the exact staged
source with its imported module path asserted. Cases include both complete tiny
legacy/recovery four-round triangle pilots, topology discrimination, full-slot
hard/soft permutation invariance, old strict checkpoint loading, matched-loss
gradients and unchanged frozen parent/decoder. Pytest cache permission warnings
only; tests themselves passed. This does not establish GPU reconstruction quality.

## Completed — 3 November 2026 14:47:51 CET

**29878344 DONE**, started14:23:10, scientific elapsed1473.94s (24.57min).
Both1,000-update phases and the full comparison finished. Finite/attached
training paths and frozen-parent checks passed. **The new branch failed to
recover reconstruction and is not promoted.** Production epoch45 is unchanged.

| Metric | Original45 | Invariant warm-up | Invariant joint pilot |
| --- | ---: | ---: | ---: |
| Exact reconstruction, 2,048 validation molecules | 99.80% | 0% | 0% |
| Chemically valid reconstructions | 99.95% | 0% | 0% |
| Random argmax validity, 10,000 latents | 75.55% | 75.55% | 75.42% |
| Random categorical validity, 10,000 draws | 63.05% | 63.05% | 60.45% |
| Identical decoded labels under 1,024 permutations | 0% | 100% | 100% |
| Mean posterior-mean L2 shift under permutation | 29.3333 | 2.818e-6 | 3.155e-6 |

All2,048 pilot reconstructions failed chemical sanitization; this is not merely
raw-label disagreement. The initial 128-molecule monitoring panel already had
0% exact reconstruction immediately after replacing the readout, before any
training, and neither short phase recovered it. That localizes the first
observed quality break to the architecture/warm-start change; it does not yet
separate initialization, representational capacity and matching/training causes.

Order sensitivity falls to floating-scale changes, but robustness alone is not
useful reconstruction. The pilot passes the declared posterior tolerance in
1020/1024 trials (99.61%); max mean/log-variance coordinate changes are2.027e-6
and4.292e-6. Do not silently call every strict numerical check passed. Decoded
labels agree in all trials but are invalid in all of them.

Random argmax validity changed -0.13pp (paired approximate95% interval
[-0.94,+0.68]pp); categorical validity changed -2.60pp ([-3.50,-1.70]pp).
These are pointwise intervals without multiplicity correction. Warm-up's exact
prior preservation is by construction with a frozen decoder. Preserved prior
generation does not validate the replacement encoder's embeddings.

Warm-up training updates averaged0.383s (matching0.244s), joint0.687s
(matching0.237s), using SciPy assignment on the GPU node. Logged final joint
matching loss fell to1.678 versus identity-order loss6.476, while reconstruction
remained invalid. No isomorphism witnesses appeared in those final logged
batches. This is a negative quality result, not a nonfinite/launch failure.

- [Readable saved comparison](../../outputs/hpc-results/vae-invariant-matched-pilot-20261103-a01/completed-summary.md)
- [Complete scientific comparison](../../outputs/hpc-results/vae-invariant-matched-pilot-20261103-a01/completed-comparison.json)
- [Native DONE and outcome receipt](../../outputs/hpc-results/vae-invariant-matched-pilot-20261103-a01/outcome-1793713688896883600.json)
- [Training/initialization/progress receipt](../../outputs/hpc-results/vae-invariant-matched-pilot-20261103-a01/status-1793713620219733000.json)
- [Hash-verified downloads](../../outputs/hpc-results/vae-invariant-matched-pilot-20261103-a01/completed-downloads.json)

No additional training or interpolation loss was started after this result.

Question: can a separate encoder/decoder branch remove input-order sensitivity
while retaining useful reconstruction and prior generation? User authorized
implementation/submission on 3 November 2026. **Epoch 45 and its production
continuation stay unchanged and paused.** No interpolation loss is introduced.

## Architecture and initialization

Use [the config](invariant-pilot.json) and
`python -m latentmol.training.invariant_pilot --config CONFIG --output NEW_OUTPUT --device cuda`.

The existing `ModelConfig.encoder_pooling` defaults to `ordered`, preserving
old checkpoint tensor names, shapes and behavior. This new branch selects
`invariant`: no atom-position parameters, shared bond-aware message passing,
then `[sum(h)/50, mean(h), occupied_count/50]` pooling. States are already
occupancy-masked; do not gate them twice. The same 320-wide residual pooled MLP
and 256-dimensional Gaussian heads remain. Decoder architecture, four triangle
rounds, atom/bond/charge vocabularies and shared graph schema are unchanged.

Copy every compatible epoch-45 tensor, including the entire decoder and GNN.
Initialize the changed readout by averaging the old position-specific weights;
mean/count feature columns start at zero. Remove only the old position tensor.
This does not preserve the old encoder function. Record copied/reinitialized
keys and parameter counts. Use new optimizer state, not a claimed exact resume.

## Bounded training protocol

Select **65,536 real training molecules**, excluding validation/test splits.
Batch64, FP32, seed11034601; shuffle all 50 input slots independently, including
padding, while applying the same bijection to charges and both bond axes.
Targets remain the original complete molecular graphs. Two phases:

1. **1,000 encoder-only warm-up updates**, decoder frozen. Deterministic latent
   means, matching-aware reconstruction plus MSE to the frozen parent encoder's
   canonical means (weight1) and log-variances (weight0.1). Encoder AdamW LR1e-4.
2. **1,000 joint updates**, stochastic VAE reconstruction and KL (beta0.001).
   Encoder LR1e-4, decoder LR1e-5, weight decay1e-5. Add a **0.25-weight frozen
   decoder KL anchor on64 fresh N(0,I) points/update** to protect learned prior
   generation while the new encoder adapts. This is distribution preservation,
   not a chemistry objective. Teacher parameters remain frozen.

There are128,000 real graph views, not additional full-dataset epochs. This
architecture pilot deliberately omits the parent campaign's costly chemistry
repair/composition/occupancy/reference losses and its real/random chemistry
ratio. The joint random points only anchor to the old decoder; they do not
receive chemical repair targets. This is a new bounded protocol, not adoption
of a different loss into the parent run. No off-target/binding model is used.

The prior anchor averages true categorical KL over atom, charge and bond
distributions. Occupancy and scalar per-pair weights come only from the frozen
reference. Do not apply category weights inside a KL: that can create a gradient
even at equal decoders. The local test checks the identity anchor's zero gradient.

Save immutable checkpoints every250updates and at both phase endpoints. Record
loss components, pre-clip gradient norm, whether clipping occurred, matching
backend/time/assignments and health panels every250updates. Clip norm5 as a
routine training safeguard; finite/attached paths and root architecture changes
are checked separately. Quality drops are findings, not automatic promotion or
abort conditions. Nonfinite computations/updates, detached required paths,
unexpected frozen-model writes and output collisions stop the affected pilot.

## Matching-aware reconstruction

Implementation: [training/matching.py](../../src/latentmol/training/matching.py).
Find one full-slot bijection for atom types, charges and **both** bond axes,
including NO_ATOM padding. Identity is the incumbent. A node/typed-degree
assignment and two rounds of bond-context assignment propose alternatives.
A bounded hard-label graph-isomorphism witness helps with symmetric graphs;
it never repairs labels or calls RDKit in the training loss. SciPy assignment
is used if installed, otherwise the included NumPy Hungarian solver is used;
the backend is recorded and no cluster environment installation is required.

Score each proposal with the actual reconstruction CE contributions using the
same **global batch denominators**, class weights, occupied-charge mask and
unordered occupied-pair mask as the existing loss. Keep a map only when that
objective improves; the size term is unchanged by permutation. Descriptor costs
are proposal heuristics, not extra training losses. Matching is detached and
ordinary CE backpropagates through the live decoder/encoder. This is bounded
approximate graph matching, not exact minimization over every permutation or a
claim that approximate alignment always makes the loss permutation invariant.

## Matched evaluation and interpretation

Compare **original epoch45**, **invariant warm-up**, and **invariant joint pilot**
on the same hardware, batches and saved panels:

- **2,048 held-out validation reconstructions**: canonical identity, raw labels,
  validity, connectivity, diversity and sulfur composition. Raw slot CE is a
  diagnostic; exact graph identity is the principal reconstruction measure.
- **10,000 fresh N(0,I) points**, five common seed blocks, argmax and one
  categorical draw each. Preserve invalid/duplicate outputs. Warm-up's random
  generation is preserved by the identical decoder, not evidence of improvement.
- **256 validation molecules × four full-slot permutations**. Record canonical
  and reordered reconstruction accuracy, valid-identity stability, exact output
  labels, and mean/max posterior-mean/log-variance changes. Floating tolerance
  rtol1e-5/atol1e-6 is reported, not silently widened. Robustness alone can be
  perfect for a constant encoder; interpret alongside reconstruction/diversity.
- The shared comparison also retains a small local-neighborhood panel,128 real
  and128 prior anchors × two directions at0.05; these are evaluation only.

The final test split remains untouched. The changed readout is trained briefly,
so poor reconstruction is an adaptation/capacity finding, not proof that an
invariant architecture cannot work. Same-lineage comparisons and this changed
loss/sampling/parameterization are not a matched full-training architecture
study. Main readouts are `summary.md`, `summary.json`, `comparison/summary.json`
and per-stage health/raw graphs; `training-completed.json` distinguishes finished
training from a later evaluation failure. New checkpoint metadata uses pilot
epoch0 and parent_epoch45, not a claim that production epoch46 completed.

## Execution

Reuse the existing node-local source launcher and GPU-routing shell with the
explicit module `LATENTMOL_EVALUATION_MODULE=latentmol.training.invariant_pilot`.
No nested login shell, new campaign controller, dependency callback or automatic
production restart. Planned resource budget: one exclusive32GB V100, four CPU
slots with4GB/slot reservation, four-hour wall limit. The first update logs
provide actual matching/training timing; no GPU speed claim is established yet.

Local checks pass 29 cases, including complete tiny four-round triangle
warm-up/joint/evaluation, encoder invariance under hard/soft input permutations,
old checkpoint compatibility, Hungarian optimality on small costs, symmetric
graph matching, exact batch-loss improvement, live gradients, frozen reference
checks and prior comparison/interpolation regressions. All 29 cases passed
again against the exact packaged snapshot; actual CLI help, shell syntax and
the submitted JSON configuration validated locally. These do not establish
native GPU quality or final reconstruction capacity.

## Submitted — 3 November 2026 14:03 CET

**Job 29878344 accepted, initially PEND on `gpuv100`.** Native LSF confirms
four CPU slots, 4 GB per slot resource reservation, four-hour limit and
`num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=30G`. No dependency
callback or automatic production continuation is attached. The job uses direct
argv through the existing node-local launcher; source and actual launch command
are saved. Other jobs, sealed sources and epoch45 remain untouched.

Archive SHA:
`a7f644e7d6568a77877623bae599249e34d85e2ea5eb45ab98b155c7d0065101`.
Read-only HPC checks verified the parent checkpoint, data manifest and pinned
GPU-routing utility, and confirmed the requested training subset fits the
1,688,539-molecule training split. Uploaded source/launcher bytes were checked
before submission. Native training/quality results are pending.

- [Source snapshot and allocation](../../outputs/hpc-submissions/vae-invariant-matched-pilot-20261103-a01/build.json)
- [Read-only input checks](../../outputs/hpc-submissions/vae-invariant-matched-pilot-20261103-a01/preflight.json)
- [Submission and exact native command](../../outputs/hpc-submissions/vae-invariant-matched-pilot-20261103-a01/submission.json)
- [Staged local fixture results](../../outputs/implementation-checks/vae-invariant-pilot-a01.xml)

Remote output:
`/work3/s234818/latentmol/runs/vae-invariant-matched-pilot-20261103-a01`.
Home control/logs:
`/zhome/95/e/206121/dbmol-control/vae-invariant-matched-pilot-20261103-a01`.
This branch alone is authorized to train; the production continuation stays
paused and no interpolation objective has been added.
