# Does epoch 45 interpolate coherently between chemical neighbors?

## Slot-aligned, gradient-capped pilot

The [follow-up config](epoch45-slot-aligned-interpolation-pilot.json) tests
whether verified intermediate targets can improve held-out paths while
preserving random-sample validity. Both arms start from the unchanged epoch-45
checkpoint, with 200 joint updates, 32 matched real reconstruction molecules
per update, and the same sampled prior anchor. The treatment adds two verified
midpoint targets per update. Both endpoint maps must be identity permutations
in the intermediate's canonical slot frame. Training uses observed molecules;
held-out derived endpoints must pass the same alignment and split checks.

The treatment keeps a maximum interpolation weight of 0.1 and lowers it on
each update as needed so the weighted interpolation gradient L2 norm is at
most 25% of the base gradient norm across the trainable parameters. The
base objective combines reconstruction, VAE KL and the frozen prior anchor.
The shared optimizer clip follows the cap. Logs record both gradient norms
and the applied weight. This bounds the gradient contribution before AdamW.

Both arms decode the same 1,024 fixed prior latents at updates 0, 25, 100 and
200, using argmax and shared categorical draws at temperature 1. Readouts
separate chemical validity from strict schema compatibility. Each observation
saves paired outcomes for its step, and the final evaluation retains the
2,048-molecule reconstruction and separate 10,000-latent panels from the
previous study. The matched arms test the combined changes in slot filtering
and gradient scaling. The full training campaign uses its existing recipe.

```bash
python -m latentmol.training.interpolation_pilot \
  --config experiments/vae/epoch45-slot-aligned-interpolation-pilot.json \
  --output NEW_OUTPUT --device cuda
```

Fifteen focused local CPU tests passed across triplet selection, gradient
scaling, fixed-prior observation and the matched runner. DTU accepted
**29899706** on 5 November 2026 as a single sequential job on `gpuv100` with
one 32 GB V100, four CPU cores and a one-hour allocation.
[Submission receipt](../../outputs/hpc-submissions/protenix-latent-editability-20261105a/slot-aligned-interpolation/submission.json).

### Slot-aligned pilot results

Both arms and final evaluation completed in about 10 minutes. The parent
checkpoint passed the unchanged-state check. The run selected 128 observed
training triplets and 64 held-out triplets with matching slots. The cap
activated on 195 of 200 treatment updates, with a median applied weight of
0.00709. Median pre-clip gradient norms were 0.221 and 0.235 for control
and treatment. The early shared panel showed
equal validity counts at updates 0 and 25. The largest treatment deficit
was seven categorical samples out of 1,024 at update 100. At update 200,
the treatment gained two argmax-valid samples and lost five categorical-valid
samples net against the control.

| Metric | Epoch 45 | Matched control | Capped interpolation |
| --- | ---: | ---: | ---: |
| Exact reconstruction | 2,044/2,048 | 2,044/2,048 | 2,044/2,048 |
| Prior argmax validity | 75.55% | 75.23% | 75.10% |
| Prior categorical validity | 63.05% | 62.92% | 62.90% |
| Valid midpoint | 59/64 | 59/64 | 59/64 |
| Exact target at midpoint | 32/64 | 31/64 | 32/64 |
| Valid at all three interior points | 54/64 | 54/64 | 53/64 |

On 10,000 shared final prior vectors, the paired changes were −0.13
percentage points for argmax (approximate 95% interval −0.31 to +0.05)
and −0.02 for categorical sampling (−0.20 to +0.16). Both arms reproduced
the same 2,044 reconstruction molecules. The pilot avoided the earlier large
validity drop and showed no clear held-out path benefit. Slot filtering,
gradient scaling and a shorter budget changed together between pilots.
[Pilot summary](../../outputs/hpc-results/protenix-latent-editability-20261105a/slot-aligned-interpolation-29899706/summary.json)
and [paired evaluation](../../outputs/hpc-results/protenix-latent-editability-20261105a/slot-aligned-interpolation-29899706/comparison.json).

## Verified graph intermediates

The [graph-target pilot](epoch45-graph-interpolation-pilot.json) compares
reconstruction training with the same training plus supervision from
verified A → B → C triplets. Both arms start from epoch 45 and take
400 updates with 32 real reconstruction examples per update. The treatment
uses two triplets per update and a 0.1 auxiliary loss weight. It decodes
the midpoint of the posterior means of A and C and applies the usual
reconstruction loss to the canonical graph of B.

The training triplets use observed MOSES graphs. Each endpoint differs
from B by one supported atom or bond label in a verified atom mapping, and
the two edits touch disjoint slots. The held-out panel uses separate MOSES
validation centers and verified endpoints. Derived endpoints may fill the
panel and are recorded separately from observed molecules. All held-out
identities are screened against training and reserved test membership.

Readouts include exact B recovery and distinct valid hard edits at 25%,
50% and 75%, exact endpoints, held-out reconstruction, and random-prior
argmax and categorical validity. The full optimization question requires
a later scorer evaluation on the candidate checkpoints.

```bash
python -m latentmol.training.interpolation_pilot \
  --config experiments/vae/epoch45-graph-interpolation-pilot.json \
  --output NEW_OUTPUT --device cuda
```

DTU job **29899015** was submitted on 5 November 2026 to `gpuv100` with
one 32 GB V100, four cores and a one-hour limit. Initial status was pending.
Eight focused local tests passed. A local check against the MOSES identity
metadata filled 128 observed training triplets and 64 held-out triplets.
All 128 held-out endpoints in that check were derived. The allocated
runner also checks the observed graphs against the stored canonical labels.

At 14:34 CET both training arms and the held-out paths had finished.
The control and treatment had 48/64 and 40/64 valid midpoints respectively,
12/64 and 11/64 exact intermediate graphs, and 39/64 and 31/64 paths valid
at all three sampled interior positions. Among the 32 triplets with both
endpoints reproduced by both arms, midpoint validity was 26/32 and 21/32,
with four exact intermediates each. All evaluated triplets had derived
endpoints. Reconstruction and prior evaluation were still running.
[Interim path evidence](../../outputs/hpc-results/protenix-latent-editability-20261105a/graph-interpolation-29899015/path-status.json).

### Final reconstruction and random-sample results

Job **29899015** completed in 687 seconds. Both arms used the same 2,048
reconstruction molecules and 10,000 prior latents. The random-sample
columns use chemical validity before a schema round-trip check.

| Metric | Epoch 45 | Control | Verified-graph interpolation |
| --- | ---: | ---: | ---: |
| Exact reconstruction | 2,044/2,048 | 2,044/2,048 | 2,040/2,048 |
| Valid reconstruction | 2,047/2,048 | 2,047/2,048 | 2,043/2,048 |
| Prior argmax validity | 75.55% | 75.62% | 68.99% |
| Prior categorical validity | 63.05% | 63.27% | 52.51% |

The treatment lost four exact reconstructions and 6.63 and 10.76
percentage points of prior validity compared to the control. The paired
95% intervals for the prior differences were −7.51 to −5.75 percentage
points for argmax and −11.71 to −9.81 for categorical samples. These
intervals describe the fixed checkpoints on this panel. The midpoint
validity decline also remained. This pilot supports deferring the loss.

[Final result](../../outputs/hpc-results/protenix-latent-editability-20261105a/graph-interpolation-29899015/completed-summary.json)
and [paired evaluation](../../outputs/hpc-results/protenix-latent-editability-20261105a/graph-interpolation-29899015/comparison.json).

### Failure analysis

The weighted midpoint loss exceeded ordinary reconstruction on 376/400
updates, with a median per-update ratio of 47.7. The median pre-clip
gradient norm rose from 0.222 to 4.260. Both arms used a soft-output KL
anchor to epoch 45 on random prior points. In the last 100 updates, the
median anchor KL was about 174 times higher in the interpolation arm.

On the identical 10,000 prior vectors, 1,368 argmax graphs lost validity
and 705 gained it. Of the losses, 857 changed only bond labels, with the
same occupied slots, elements and formal charges. Valence failures rose
from 980 to 1,652 and kekulization failures from 325 to 714, while
disconnections fell from 1,133 to 735. Reconstruction at encoded test
molecules remained near the starting level.

The triplet builder verified edits with atom maps. Training still encoded
each endpoint and supervised the intermediate in its own canonical slot
order. All eight net midpoint validity losses came from held-out triplets
with a nonidentity map. The saved results establish scale imbalance and
slot-order sensitivity. Their separate effects on prior validity remain
unmeasured.

[Training audit](../../outputs/hpc-results/protenix-latent-editability-20261105a/graph-interpolation-29899015/training-mechanism-audit.json)
and [paired graph audit](../../outputs/hpc-results/protenix-latent-editability-20261105a/graph-interpolation-29899015/prior-failure-audit.json).

## Atom-mapped training pilot

DTU accepted job **29890181** at 11:41:53 CET on 4 November 2026. Both arms run
sequentially on one V100 with four CPU slots and a four-hour allocation limit.
The 12:05 CET scheduler check still shows PEND while the job waits for its
physical GPU reservation. The 23 local checks passed on the staged source.
[Scheduler status](../../outputs/hpc-results/vae-atom-mapped-interpolation-20261104-a01/status-1793790352190592800.json).
[Submission](../../outputs/hpc-submissions/vae-atom-mapped-interpolation-20261104-a01/submission.json),
[source and budget](../../outputs/hpc-submissions/vae-atom-mapped-interpolation-20261104-a01/build.json).

[Configuration](epoch45-interpolation-pilot.json) starts two independent arms
from the same epoch 45 weights with fresh AdamW state. Each receives 400 joint
updates on the same 12,800 real training rows, batch size 32 and matched sampling
seeds. Both use canonical cached inputs, reconstruction, KL at 0.001 and a
frozen-decoder prior anchor at 0.25. The treatment adds a mapped interpolation
cross-entropy at weight 0.1, two training pairs per update and all nine interior
fractions. Encoder and decoder learning rates are both 1e-5.

Select up to 192 training pairs from 30,000 training rows and up to 128
validation pairs from the validation split, using the same graph matcher and
eligibility rules as below. For each pair, encode both endpoints in their
canonical order. At 10%, 20%, through 90% weight of B, linearly mix the
posterior means and supervise the decoded atom, charge and bond distributions
with the same mixture of mapped hard labels. Score each prediction in both
endpoint slot frames and backpropagate through the lower cross-entropy.
The loss averages the separately normalized atom, charge and bond terms.
Padding and pairs with no bond at either endpoint have reduced row weights.
Weights apply to whole distributions, preserving the requested probabilities
for bond transitions. The graph mixture is probability supervision, with
chemical validity measured separately on decoded graphs.

Evaluate epoch 45 and both final checkpoints on identical held-out pairs
at all nine fractions and both endpoints, with argmax and four categorical
draws per point. Save raw graphs, endpoint identity, mapped target JS,
and paired argmax validity. The shared checkpoint comparison also measures
2,048 reconstructions, 10,000 random latents and shuffled-input robustness.

```sh
python -m latentmol.training.interpolation_pilot --config experiments/vae/epoch45-interpolation-pilot.json --output NEW_OUTPUT --device cuda
```

The comparison isolates the added objective at equal base data and update
count. The treatment decodes 18 additional latent vectors per update.

### Matched pilot results — 4 November 2026

DTU job **29890181 DONE**. Both 400-update arms and evaluation finished
in about 14 minutes. The parent checkpoint hash was preserved. Evaluation
used 128 held-out pairs and all 128 retained exact endpoint graphs and
labels in every arm.

| Metric | Epoch 45 | Control | Interpolation |
| --- | ---: | ---: | ---: |
| Valid midpoint, argmax | 66/128 | 68/128 | 64/128 |
| All nine interior argmax graphs valid | 44/128 | 47/128 | 44/128 |
| Mean interior target JS, lower is closer | 0.02137 | 0.02140 | 0.02002 |
| Exact reconstruction | 2,044/2,048 | 2,044/2,048 | 2,043/2,048 |
| Prior argmax validity | 75.55% | 75.62% | 75.43% |
| Prior categorical validity | 63.05% | 63.27% | 61.06% |

The treatment matched the soft interpolation targets more closely, while
observed hard path validity and categorical prior validity fell against
the matched control. Midpoint gains and losses were three and seven pairs,
with a paired 95% interval of about -8.0 to +1.7 percentage points. The
pilot did not satisfy the planned condition for the full continuation.
[Verified summary](../../outputs/hpc-results/vae-atom-mapped-interpolation-20261104-a01/completed-summary.json).

### Midpoint mechanism and chemistry follow-up

The midpoint cross-entropy target is a collection of categorical marginals.
Taking argmax of the exact target marginals gave **42/128** valid graphs
across the held-out pairs. Independent sampling of those same marginals
gave **2,861/4,096** valid draws. A single shared A-or-B draw yields one of
the two verified valid endpoints. These controls isolate the loss of
coordination across graph labels. Five of the seven midpoints that became
invalid under training still moved closer to the known target by JS.
Those seven failures were three valence errors and four kekulization
errors. [Midpoint diagnostic](../../outputs/hpc-results/vae-atom-mapped-interpolation-20261104-a01/midpoint-target-diagnostic.json).

A local replay of the seven lost midpoints used their saved latent vectors
and the verified frozen interpolation checkpoint. The bounded chemistry
search found a valid bond repair for all seven. Every point received a
finite, nonzero latent gradient. One chemistry-only latent step of length
0.03 made four of seven raw argmax graphs valid. This tests the local repair
signal at held-out points, separate from training or generalization.
[Repair replay](../../outputs/hpc-results/vae-atom-mapped-interpolation-20261104-a01/lost-midpoints-repair-smoke.json).

DTU **29892811** is a three-arm comparison with the original 400 updates,
same epoch-45 start, real rows, pair schedule, seeds, nine fractions and
evaluation. It repeats reconstruction-only and mapped interpolation
controls, then adds atom-preserving repair NLL to the mapped points at
weight 0.25 in the third arm. The
[chemistry config](epoch45-interpolation-chemistry.json) uses the existing
structural proposal search and fixed-target NLL. The teacher changes bonds
while keeping slot occupancy, elements and charges fixed, with six edits
at most for a structural proposal. Already-valid points and searches
without a verified witness give no repair gradient. The mean includes
every attempted point. Logged status counts distinguish accepted,
already-valid and censored attempts.

The isolated comparison tests chemistry at the interpolation points.
The existing prior anchor and reconstruction/KL remain shared across
all three arms. Held-out outcomes include path validity, exact endpoints,
valid interiors distinct from both endpoints, MOSES reconstruction and
prior validity. Twenty-six local fixtures passed, including a disconnected
positive fixture and the three-arm training/evaluation entry point.
[Submission](../../outputs/hpc-submissions/vae-interpolation-chemistry-20261104-a01/submission.json).
The full continuation stays paused pending the matched outcome.

**Chemistry follow-up completed.** Job **29892811** ran all three 400-update
arms and evaluation in 18.2 minutes. All 128 endpoint pairs retained exact
graphs and labels in every arm.

| Metric | Control | Interpolation | Interpolation + chemistry |
| --- | ---: | ---: | ---: |
| Valid midpoint | 68/128 | 64/128 | 66/128 |
| All nine interior argmax graphs valid | 47/128 | 44/128 | 46/128 |
| Valid midpoint distinct from both endpoints | 39/128 | 37/128 | 38/128 |
| Exact reconstruction | 2,044/2,048 | 2,043/2,048 | 2,044/2,048 |
| Prior argmax validity | 75.62% | 75.43% | 75.54% |
| Prior categorical validity | 63.27% | 61.06% | 61.41% |

Chemistry rescued two midpoint graphs relative to interpolation alone, with
zero additional losses on that paired panel. The approximate paired 95%
interval spans -0.59 to +3.72 percentage points. The combined arm did not
exceed the reconstruction control on the measured path or generation
outcomes. Keep interpolation as an evaluation and defer its training loss
in the larger data run.
[Verified follow-up summary](../../outputs/hpc-results/vae-interpolation-chemistry-20261104-a01/completed-summary.json).

### Local CUDA feasibility check — 4 November 2026

Four updates per arm completed on the laptop RTX 4060 with the authenticated
epoch-45 weights, unchanged FP32 loss code, batch size 32, 32 prior anchors
and two pairs at nine fractions. This used repeated eight-atom fixture
graphs and an identity atom map to measure memory and runtime. The control
peaked at 7,909 MiB allocated and took about 1 second per warm update.
Interpolation peaked at 10,279 MiB allocated on an 8,188 MiB GPU and took
roughly 11 seconds per warm update. This is consistent with Windows using
shared system memory. The timings imply roughly 80 minutes for both full
training arms, plus input preparation and evaluation. No local scientific
comparison was run. A faithful local repeat also requires the 4.3 GiB
dataset cache. Local RDKit 2024.09.6 matches its manifest.

### Interactive allocation check — 4 November 2026

DTU rejected the one-hour request to `gpuvoltasxm2i` with
`walltime not supported`. The default-time retry obtained an idle 32 GB
V100 as job **29891146**, with a scheduler limit of 15 minutes. That job
was stopped during input setup, before either training arm started.
The handoff held **29890181** before any interactive worker startup and
then resumed it after the interactive worker exited. The batch job was
verified pending and remains the sole submission for the full comparison.

## Completed — 3 November 2026 12:40:24 CET

**Job 29877725 DONE**. Scientific evaluation took **124.62 s**; 128 disjoint
pairs qualified from the 20,000 validation molecules. The pool produced only
146 candidate edges (140 examined), not enough to reach 200 pairs under the
fixed rules. One MCS timed out. No thresholds or endpoint filters were relaxed.
All frozen-state/checkpoint checks pass; training remains paused.

| Weight of B | Canonical argmax valid | Matched argmax valid | Canonical categorical valid | Matched categorical valid |
| --- | ---: | ---: | ---: | ---: |
| 25% | 127/128 (99.22%) | 128/128 (100%) | 98.63% | 99.80% |
| 50% | 66/128 (51.56%) | 126/128 (98.44%) | 44.53% | 98.24% |
| 75% | 127/128 (99.22%) | 124/128 (96.88%) | 98.24% | 96.00% |

**The matched-arm validity is confounded by endpoint reconstruction.** All 128
canonical endpoint pairs reconstruct exactly, including labels. After
re-encoding reordered B, only 38/128 pairs retain both correct endpoint
identities and 37/128 retain exact labels. The other valid matched endpoints
often represent a different molecule; high validity alone is not success.
The exact-label-endpoint subset has 37/37 valid argmax outputs at each interior
fraction, but this is a small conditioned subset, not the primary success rate.

Saved-graph analysis finds that all **30 identity-map pairs** have valid
canonical midpoints. Among the **98 pairs needing reordering**, only 36 have
valid canonical midpoints. The 62 canonical midpoint failures are 58
sanitization and four disconnected graphs. This supports investigating slot
ordering as a contributor; the groups also contain different molecules and
do not establish isolated causality. Median canonical endpoint latent distance
is 4.884, matched distance 2.876, and permutation-induced B shift 4.725.
Latent metric distances alone do not establish semantic smoothness.

The standalone HTML, full summaries, pairs and raw outputs were downloaded
with native/local byte counts and SHA checks. No models were rerun for this
saved-result analysis.

- [Interactive report (21.3 MiB)](C:/Users/nikol/DBMol_Local/reports/vae-epoch45-interpolation-20261103-a01/index.html)
- [Complete scientific summary](../../outputs/hpc-results/vae-epoch45-interpolation-20261103-a01/completed-summary.json)
- [Native terminal/selection/results receipt](../../outputs/hpc-results/vae-epoch45-interpolation-20261103-a01/status-1793706037699498600.json)
- [Saved-graph ordering analysis](../../outputs/hpc-results/vae-epoch45-interpolation-20261103-a01/saved-graph-analysis.json)
- [Verified downloads](../../outputs/hpc-results/vae-epoch45-interpolation-20261103-a01/downloads.json)

User authorized implementation/submission on 3 November 2026. This is frozen
checkpoint evaluation, not interpolation training. VAE training stays paused;
there is no optimizer, repair, continuation callback or automatic promotion.

## Pair selection and atom correspondence

[Config](epoch45-interpolation.json): target **200 disjoint pairs** from the
20,000-molecule validation split. The final test split stays untouched. Require
the same heavy-atom count (8–50), identical Murcko scaffold (empty is allowed
for acyclic pairs), Morgan radius-2/2048-bit Tanimoto >=0.65, and an element/
bond-preserving common core covering >=80% of the atoms. Limit each scaffold
to 10 pairs. Up to 1,000 candidates, one-second MCS timeout each; timeouts are
recorded and skipped. A smaller cohort is reported without relaxing thresholds.
Selection uses chemistry only, before any decoder output is observed.

The reusable [pair matcher](../../src/latentmol/evaluation/molecular_pairs.py)
anchors an MCS with ring/order constraints. At most six unmatched atoms are
assigned by minimum additive local-environment cost. Symmetric embeddings and
alignments have explicit caps; prefer fewer label edits, then more unchanged
slots. This is a bounded heuristic, not a unique/global-optimal atom mapping.
Accept at most two element, two charge and four unordered bond-label changes.
Save the full B-original-index-for-A-aligned-slot permutation and core matches.
Verify source cache labels against RDKit atom ordering and chemical identity
after permutation; no label repair or chemical transformation is performed.

## Encoding-sensitive control

The current encoder is **not permutation invariant**. Evaluate two arms:

- **canonical:** interpolate `E(A)` and `E(B)` in their original canonical slots.
- **matched:** reorder the complete B input graph into the matched slots, then
  encode it again. Interpolate `E(A)` and `E(permuted B)`; targets and decoded
  endpoints use that same ordering. Never change only B's target while silently
  retaining its canonical latent vector.

Use deterministic encoder means and straight linear interpolation. Evaluate
the requested **25%, 50%, 75% weight of B**, plus both endpoints. At every point
retain argmax and **eight categorical draws**, temperature 1, common seeded RNG
across arms. Up to **18,000 hard graphs** including endpoint/control attempts.
All raw invalid and duplicate graphs remain in the denominator and saved files.

Primary summaries include **all pairs**. Separate diagnostic views condition on
both endpoints reconstructing the correct graph, and more strictly on both
reconstructing the exact aligned labels. Poor reordered-endpoint reconstruction
can confound matched-path failures; do not call that latent nonsmoothness.
Endpoint identity-map pairs, latent distances and permutation-induced shifts
are reported. Similarity here selects chemical, not latent, neighbors.

## Readouts

Report chemical/representation validity separately, connectivity, composition/
sulfur, unique valid identities, fingerprint similarity to each endpoint and
the fraction different from both endpoints. This last metric does not establish
training-set novelty or useful intermediate properties. A valid hard decoder
output need not vary continuously because categorical labels change discretely.

Measure normalized Jensen–Shannon probability distance to (a) the weighted known
graph labels and (b) the weighted decoded endpoint distributions. The latter
helps separate interpolation curvature from existing endpoint reconstruction
errors. Use raw conditional bond probabilities consistently; report occupied
atom/charge terms, all occupied pairs, union-of-endpoint bond edges and padding
separately. The equal-term summary uses atom/charge/union-bond terms; NONE pairs
cannot swamp it. This is **not** distance to a chemically valid molecule.
Save adjacent-point JS, entropy, expected atom counts and latent norms too.

Categorical draws share their pair/latent; uncertainty clusters by pair instead
of counting eight draws as independent molecules. Intervals are descriptive,
pointwise and unadjusted. This selected-cohort test cannot prove global latent
smoothness, chemical validity everywhere, synthesis feasibility or binding.

## Commands, outputs and budget

```sh
python -m latentmol.evaluation.interpolation \
  --config experiments/vae/epoch45-interpolation.json --output NEW_OUTPUT --device cuda

# Regenerate only the presentation from saved scientific outputs:
python -m latentmol.reporting.interpolation SAVED_RUN
```

The renderer refuses to overwrite an existing HTML report; preserve it before
generating another. It never loads an encoder, decoder or external scorer.

Use one exclusive V100, four CPU slots, 4 GB per slot reservation, **two hours**.
The shared evaluation shell uses `LATENTMOL_EVALUATION_MODULE=latentmol.evaluation.interpolation`;
its default comparison entry point is unchanged. Existing node-local source
launcher and pinned GPU-routing utility are reused. No existing job, callback,
sealed release or checkpoint is edited.

Outputs: `pairs.json`, `endpoint-latents.npz`, `summary.json`, `progress.json`,
per-arm raw gzip JSONL, probability metrics and first-16-pair probability
snapshots, plus standalone **`report.html`** with a pair selector, both encoding
arms, five molecule snapshots and expandable raw invalid graphs. Nonfinite,
incompatible inputs, changed frozen weights and output collisions stop the
affected run; quality failures remain observations. Presentation failure is
recorded separately after scientific results are saved.

Local result: **11 tests pass**, including the prior comparison fixtures plus
six interpolation tests (matching, permutation identity, slot-sensitive
re-encoding, interpolation arithmetic/JS, complete frozen execution and saved
report, collisions and pair-budget findings). Native GPU results are pending.

## Submitted — 3 November 2026 12:38 CET

**Submission/start record: 29877725 RUN since 12:38:08 CET on `gpuv100`.** One exclusive GPU, four
CPU slots with 4 GB per slot reservation, two-hour wall limit. Native LSF
confirms `num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=16G`.
No training callback or restart is attached. Existing jobs remain untouched.

Exact packaged source SHA:
`11641ab008c2e996ccdd4cb3b098e5388ae934687f1c8149e83456822b565724`.
All 11 fixtures passed again against that staged snapshot; CLI help and shell
syntax pass. Read-only HPC checks authenticate epoch 45, the data manifest
and GPU-routing helper. Uploaded archive and node-local launcher are verified
before `bsub`. The actual job uses direct argv, avoiding the earlier nested
login-shell launch failure in the checkpoint comparison.

- [Source and allocation](../../outputs/hpc-submissions/vae-epoch45-interpolation-20261103-a01/build.json)
- [Asset checks](../../outputs/hpc-submissions/vae-epoch45-interpolation-20261103-a01/preflight.json)
- [Submission and exact scheduler command](../../outputs/hpc-submissions/vae-epoch45-interpolation-20261103-a01/submission.json)
- [Local staged fixtures](../../outputs/implementation-checks/vae-epoch45-interpolation-a01.xml)

Remote outputs:
`/work3/s234818/latentmol/runs/vae-epoch45-interpolation-20261103-a01`.
Home control/logs:
`/zhome/95/e/206121/dbmol-control/vae-epoch45-interpolation-20261103-a01`.
At this submission/start snapshot, native quality results were pending.
VAE training stays paused. At the initial startup check the Python module and allocated GPU routing passed;
input/model preparation had not yet written its ready marker. This confirms
startup only, not completed inference or quality checks.
[Native startup receipt](../../outputs/hpc-results/vae-epoch45-interpolation-20261103-a01/status-1793705944579181200.json).
