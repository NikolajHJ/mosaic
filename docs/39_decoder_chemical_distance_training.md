# Matched decoder fine-tuning with probability-aware chemical regularization

Status, 8 October 2026: the user explicitly authorized a decoder-training
comparison using a rule-based, probability-aware measure of changes needed
for chemical validity, with random-latent generation as a primary evaluation.
Source070 is sealed and uploaded. CPU engineering job **29622690 DONE**
passed **88 checks with one CUDA-only check skipped**, in 14.23 seconds.
Old native GPU **29622699 was retired after preserving the first control's
authenticated epoch7 checkpoint**. Observed pace exceeded its sealed
three-hour case budget. User-authorized budget recovery is running in
[docs41](41_decoder_budget_recovery.md): source073 attempt002 is uploaded and
CPU 29624915 DONE (112 passed/two CUDA skips). **All three seed 17 ten-epoch
arms and endpoint evaluations are complete.** Both chemistry arms improve
raw prior validity and unique yield versus original/control under the saved
exploratory paired-panel criteria, retaining held-out reconstruction within
the one-percentage-point tolerance. At **09:33 DTU-local**, seed 23 GPU 29624920
is RUN: control complete, coefficient 0.1 has four saved epochs, coefficient 1
has not started. Replication and the full two-seed comparison remain pending.
This is **local valence/connectivity regularization**, not search-target NLL.
Most random outputs remain invalid even at the strongest tested coefficient.
DeFoG is not required for this experiment. Existing source047 checkpoints and
the source064/source065/source069 molecular-optimization experiments remain
unchanged historical artifacts and controls.

## Verified seed17 endpoints, 8 October 2026

All trained rows use ten completed epochs on 215,000 graphs, the same frozen
encoder, protected split and independent prior panel. Raw graphs are scored
without repair replacement or invalid filtering. Argmax uses 4,096 points;
sampling uses four draws per point, 16,384 attempts. Held-out reconstruction
uses deterministic posterior means on 7,347 graphs.

| Decoder | Valid argmax / 4,096 | Valid draws / 16,384 | Unique valid draws | Valid reconstruction / 7,347 | Exact reconstruction / 7,347 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untouched original | 64 | 124 | 118 | 7,220 | 7,160 |
| Seed 17 matched control | 65 | 130 | 122 | 7,234 | 7,174 |
| Seed 17 chemistry coefficient 0.1 | 251 | 536 | 522 | 7,236 | 7,166 |
| Seed 17 chemistry coefficient 1 | 461 | 1,126 | 1,105 | 7,235 | 7,154 |

The strongest arm gives **11.25% valid argmax outputs** and **6.87% valid
categorical draws**. Both chemistry arms have positive paired latent-cluster
validity confidence intervals in both modes versus both references; their
saved controller assessments also retain unique-valid yield and reconstruction.
This is promising **one-training-seed evidence**, with no multiple-comparison
correction or claim about binding, synthesis or every latent point. The
untouched original remains available; second-seed chemistry outcomes and
search-guided outcomes are not established. The bounded repair diagnostic is
separate: conditional witness-cost means are not population-wide distances
or globally nearest valid molecules. Not every proxy improves: the ghost-bond
diagnostic worsens despite the raw validity and unique-yield improvements.

Verified [status evidence003](../outputs/hpc-results/source073-budget-recovery/status-evidence003.tar.gz)
SHA-256 `9bf0e47dc3584e521288adf08456b1ef51cad1e2dbb9082dcbe2714563562c19`
contains [completed seed17 cases](../outputs/hpc-results/source073-budget-recovery/extracted-status003/recovery/segment-0/cases.json),
[original evaluation](../outputs/hpc-results/source073-budget-recovery/extracted-status003/original.json),
four [checkpoint identity matches](../outputs/hpc-results/source073-budget-recovery/extracted-status003/checkpoint_identity_checks.json)
and the [09:33 scheduler snapshot](../outputs/hpc-results/source073-budget-recovery/extracted-status003/scheduler_snapshot.json).
All 29 data-file hashes in 30 members were verified with local bundled Python
3.12.14 on Windows, metadata/file checks only; no local numerical evaluation.
The ledger preserves earlier incomplete snapshots and budget failures.

This is a scoped extension of the earlier preference to use the completed
AE/VAE foundation with frozen external models. It authorizes fine-tuning the
existing molecular decoder, not training an in-house activity scorer,
chemistry reward surrogate or replacement external model. Freeze the encoder
in both training arms; freeze any resulting decoder before a later molecular
latent-optimization experiment. All numerical checks, preprocessing,
fine-tuning and evaluation remain on DTU HPC.

## Authenticated original model and data

The starting model is the same stabilized source047 VAE used by recent Boltz
latent studies: **epoch 89, best-validation-validity** checkpoint
`/work3/s234818/latentmol/runs/20261001T001617Z_vae-train_c008c090/best_validity.pt`,
SHA-256 `1893ddc183473fd4f27c6cd921d8b302f67ab388b660ada44bf3a2d6d5e28088`.
Do not overwrite it or silently substitute the epoch-86 best-reconstruction
or epoch-100 last checkpoint. Its completed original workflow is job
**29533195**, which stopped at 100 epochs under its validation plateau rule;
that stopping decision is not proof of global convergence.

| Historical identity | Recorded value |
| --- | --- |
| Dataset/cache | `/work3/s234818/latentmol/data/zinc215k-seed17-v1` |
| Train / validation / test | 215,000 / 7,347 / 7,347 molecules |
| Dataset SHA-256 | `560840389f7d85592ef8e16a5566e26b5ab3e5dfa30ebe392366dbf54489974f` |
| Split SHA-256 | `e15daf547436b66cbbd54ae82f53f153d54e2bd24edee9bf2c5787099cbe6e79` |
| Source047 archive SHA-256 | `1ad89fb80d74f5cb9a2ce06a41e399a84459e4f8cdd6255d6fcc0f2587ec7814` |
| Original source manifest | `49f44973efbca6c4c7dc72fbbc0befe3194ba3ef48927512d59ad8ca46efdd51` |
| Original workflow config SHA-256 | `a4c4c5f10d86e7aecae4066730a07fb8027404617275eb587bbd29f7f552a963` |
| Exact full-training YAML SHA-256 | `98879286d5182cc9151e8a62bb9e4375452f36811992333a3aa1aa5407017f2d` |

Architecture is latent **256**, atom/node hidden **96**, three encoder layers,
`stable_deep_pair_refine` decoder, trunk **512**, global hidden **320**, pair
hidden **48**, two refinement rounds and four attention heads. The graph
schema remains 50 heavy-atom slots; categorical real elements plus `NO_ATOM`,
five charge labels and symmetric five-class bonds. Keep schema restrictions,
round-trip rejects and protected old split memberships described in
[docs07](07_full_training_protocol.md).

The original resolved training record uses seed 17, batch 64, learning rate
0.001 with later declared reductions, weight decay 0.00001, gradient cap 5,
VAE beta 0.001 and 20-epoch historical KL warm-up. Reconstruction settings
are bond weight 2, charge weight 1, no-bond weight 0.1, padding weight 0.2
and size weight 0.1. These describe the initializer's history; the new
fine-tuning settings below have their own pending source/config identities.
A frozen encoder makes its KL contribution constant with
respect to decoder weights; do not describe that as learning a new posterior.

Historical evidence is locally available in
[stable-vae-run.json](../outputs/hpc-results/architecture-20261001-status/stable-vae-run.json),
[stable-vae-summary.json](../outputs/hpc-results/architecture-20261001-status/stable-vae-summary.json)
and [best-validity held-out evaluation](../outputs/hpc-results/architecture-20261001-status/stable-vae-best_validation_validity-test.json).
The latter records **12 valid/1,024 prescribed standard-normal draws**, all
12 unique, alongside **7,220 valid and 7,160 exact encoded-mean reconstructions
out of 7,347 held-out molecules**. Conditional reconstruction is a different
task and does not imply high random-prior generation validity. Before new training,
verify the actual checkpoint, dataset/split and compatible model state on
HPC; the prior metadata is lineage evidence, not a new preflight pass.

## Three-way comparison and matched training grid

| Evaluation arm | Training changes | Purpose |
| --- | --- | --- |
| Original | No additional updates; preserve epoch-89 checkpoint | Historical starting point evaluated under the same new evaluation protocol |
| Reconstruction plus matched anchors control, alpha 0 | Fine-tune decoder only, encoder frozen; original reconstruction plus the same occupancy/aromatic reference anchors used by regularized arms | Measure extra decoder updates and anchors without the chemical gradient |
| Chemical-regularized decoder, alpha 0.1 or 1 | Same start, frozen encoder, reconstruction and anchors, plus the chemical term | Isolate the additional chemical regularization |

All trained arms start from exact identical decoder/encoder states. Match
training graphs/order, batch counts, posterior latent-noise draws, prior-latent
draw budgets, optimizer and requested updates. Do not compare an updated
regularized decoder only to the untouched original and attribute every change
to chemistry. Record extra search/oracle CPU work separately from equal
gradient updates; equal update counts are not necessarily equal wall time.

The declared grid has **training seeds 17 and 23**, each with chemical
coefficients **0, 0.1 and 1**: **six trained cases**, plus the unchanged
epoch-89 reference. Each case requests **ten epochs over all 215,000 training
molecules**, batch **64**, a fresh decoder-only **AdamW** at learning rate
**0.00003**, weight decay **0.00001**, and gradient cap **5**. Each update
uses all 64 graphs for reconstruction, chemistry on the **first 16 posterior
points from that shuffled batch**, and **16 unfiltered standard-normal prior
points**. The smaller final batch uses its available points. Match shuffle, posterior epsilon, prior
and connectivity Monte Carlo RNG streams within each seed across coefficients;
keep validation/final evaluation RNG separate. Actual update/data counts
must be verified, especially for timeout-censored cases.

The unchanged decoder supplies detached reference occupancy and aromatic
probabilities at the same prior points. Both reference anchors have weight
**one in every trained arm**. Thus alpha-zero is a reconstruction-plus-anchors
control, not pure original-loss continuation. Evaluate every arm on the same
fixed unfiltered prior bank; no outcome-driven redraw, validity filtering or
replacement is allowed. Both control and regularized arms compute the same
chemical/Monte Carlo diagnostics and consume the same draw budgets, even
when alpha is zero.

## Differentiable chemical proxy and shared reference anchors

The training objective is a **probability-aware defect proxy**, not a nearest
chemically valid molecule distance. The authored
[decoder_regularization.py](../src/latentmol/decoder_regularization.py)
combines the unchanged reconstruction objective, frozen-reference anchors
on prior points, and alpha times the chemical term. Chemistry is applied
equally to posterior reconstruction and prior points:

```text
loss = original reconstruction
     + occupancy_reference_KL(prior)
     + aromatic_reference_KL(prior)
     + alpha * [0.5 * chemistry(posterior) + 0.5 * chemistry(prior)]
```

Any recorded frozen-encoder posterior KL is constant with respect to decoder
parameters. No learned critic, DeFoG teacher, native scorer or encoder is
trained by this objective.

The [local chemistry module](../src/latentmol/chemistry_regularization.py)
uses an explicit versioned element/formal-charge allowed-valence table,
`rdkit_book_2024_09_1_explicit_charge_v1`. It computes
**E[max(incident bond-order sum − maximum allowed valence, 0)]**, using an
exact categorical convolution with overflow mass and first moment retained.
This is an expectation of excess, not a hinge applied to mean valence that
would discard rare bad tails. Gate each neighbor by its occupancy once and
the focal atom once; use symmetric raw conditional bond probabilities.
Divide the graph's total excess by **3 × max(expected occupied slots, 1)**.
The factor three is a declared bond-order scale, not proof that this counts
the minimum number of edits; an edge can affect both endpoint valences.

Unsupported represented element/charge combinations receive a separate
probability-mass penalty, divided by max(expected occupied slots, 1).
The exact probability of an empty draw is a separate penalty. Local weights
for valence, unsupported charge and empty risk are one. **Isolated-atom
weight is zero** in this protocol because global connectivity is included;
optional ghost/size terms are also disabled. Undercoordination costs zero
because implicit hydrogen is unrepresented. **AROMATIC counts as order one**,
an optimistic lower-bound proxy; Kekulization, aromatic-ring consistency,
radicals, stereochemistry and globally compatible chemical repairs are not
established by this table. Installing a different RDKit does not silently
replace its versioned training rules.

The [global connectivity module](../src/latentmol/connectivity_regularization.py)
uses **four independent full-law hard graph draws** at each latent point,
without rejection or repair. For a draw with `K` occupied atoms and `C`
connected components, its cost is `max(C - 1, 0) / max(K - 1, 1)`.
`C - 1` is the number of unrestricted edge additions needed to connect its
components; it ignores chemical constraints on those new edges. Empty and
singleton draws have zero connectivity cost, so the separate empty risk and
common occupancy anchor remain necessary. Unlike isolated-atom counting,
this component test also detects disconnected multi-node fragments.

Connectivity gradients use the marginal Bernoulli node/edge-presence log
probability and an independent **leave-one-out baseline** from the other
draws at that same latent point. Real-element identity, charges and individual
non-NONE bond categories are marginalized for this connectivity objective.
Hard draws/component labels are detached; the derivative passes through
the stable presence log probability. It is an unbiased **first-order
score-function estimator** under the declared independent sampling law,
not a pathwise derivative through a graph or an exact deterministic gradient
from four draws. Its variance and zero-advantage frequency must be reported.
Finite differences of one fixed sampled realization do not validate that
estimator; higher derivatives have no claimed interpretation.

Reference anchors use Bernoulli **KL(original || current)**, evaluated with
stable log-softmax/logsumexp, without an epsilon floor or moving reference.
Occupancy partitions `NO_ATOM` against all real elements. Aromatic anchoring
partitions `AROMATIC` against other bond categories on unordered pairs, with
**frozen original occupancy weights**, so deleting current atoms cannot
erase the aromatic penalty. These shared anchors reduce size/aromatic
shortcuts; they neither freeze every chemical identity nor guarantee validity.

No RDKit conversion or valid-repair search enters this training forward path.
The rule proxy is differentiable; component draws are discrete with an
explicit score-function estimator. Final raw validity and bounded repair
witnesses remain evaluation tasks outside autograd.

## Independent generation and reconstruction evaluation

Validation uses **256 prior vectors, seed 81000**; final evaluation uses
**4,096 prior vectors, seed 91000**. These independent unfiltered banks are
the same for the original and all trained arms and are separate from training
RNG streams. Each latent point gets **argmax plus four categorical draws**,
with no replacement or filtering: 1,280 validation and 20,480 final candidate
memberships per evaluated checkpoint. Argmax and categorical validity/uniqueness
retain separate denominators. Shared RNG streams do not force identical
categorical graph labels when different decoder probabilities are evaluated.

Final deterministic encoded-mean reconstruction covers all **7,347 test
molecules**. The frozen encoder makes those latent means a common input
across decoder variants. Reconstruction retention and unrestricted prior
generation are separate results. Novelty is measured against the complete
215,000-molecule training split, not a small training batch.

The first **64 prior argmax outputs**, in prescribed bank order rather than
chosen for validity, receive the existing probability-aware repair diagnostic:
at most **three edits**, **256 candidate checks per depth**, **beam four**.
It uses raw conditional bond probabilities before occupancy gating and
conditioned real-element probabilities for the argmax occupied slots.
The fixed-occupancy, slot-labeled search can change atom types, charges or
unordered bond labels, not silently add/delete slots. Retain already-valid,
witness-found, censored and unsupported outcomes. Its returned best observed
edit/probability cost is an upper bound from a bounded witness search,
not globally minimal chemical distance; no witness is censored, not infinite
proof of irreparability. This search is an **evaluation diagnostic** outside
the training gradient path and does not rewrite raw generation results.
The existing metric definition/floor/likelihood limits are in
[docs27](27_graded_quality_diagnostics.md); new reports must record the actual
policy rather than inherit unstated assumptions.

Predeclare new checkpoint roles, selection rules, training cap and evaluation
seeds before results. Preserve train/validation/test assignments; test cannot
drive training decisions.
The historical holdouts have already appeared in method discussions, so
report their exposure rather than claim pristine independent prospective
validation. Use the same new evaluator/candidate quotas on the original and
both fine-tuned arms. The declared endpoint is the **last fixed epoch only**;
validation monitoring does not select a better-looking checkpoint. Timed-out
partial checkpoints are retained and excluded from a completed matched comparison.

Required evidence includes raw argmax and prescribed categorical prior
validity, unique valid identities, novelty against the complete training split,
element/charge/bond/count distributions, and decoded collapse or concentration.
Preserve invalid and unsupported denominators. Evaluate deterministic
encoded-mean reconstruction retention on protected held-out molecules separately
from unrestricted prior samples. Where bounded repair witnesses are measured,
report observed edit/probability burden and censoring; do not substitute
repaired chemistry for raw decoder performance.

Check that the frozen encoder state is identical before/after both training
arms, decoder parameters actually update, all losses/gradients are finite and
checkpoints reload with the declared schema. Record original, trained and
trainable parameter counts, batch/update counts, RNG identities, losses and
search costs. Attribute an improvement to this regularizer only relative to
its matched trained control, with sampling/search budgets declared. This
experiment alone cannot establish Boltz binding, synthesis feasibility,
selectivity or superiority of a globally smooth latent manifold.

The predeclared exploratory advantage gate requires **both argmax and sampled
raw-validity differences** to have a paired latent-cluster bootstrap 95% CI
with lower bound above zero, versus **both the unchanged original and the
same-seed matched control**. The bootstrap resamples latent IDs, keeping each
point's four draws together, with 2,000 replicates and seed 92000. Unique valid
identities per attempted candidate must not decrease in either output mode
against either reference. Test reconstruction validity and exactness may fall
by at most **one absolute percentage point** versus each reference. A proxy
decrease or cheaper repair witness alone cannot pass this gate. Report seeds
17 and 23 separately; this is exploratory finite-panel evidence with **no
familywise/multiple-comparison correction** or general chemical guarantee.

## Engineering smoke and bounded serial HPC study

Before full fine-tuning, the declared engineering smoke runs two epochs on
**512 training graphs**, **64 validation graphs** and **32 prior-evaluation
points**, comparing alpha **0 and 1** with the same training seed. Its gate
requires finite losses/gradients, frozen encoder parameters and buffers,
actual decoder updates and matched data/noise/Monte Carlo RNG. It saves
checkpoint receipts; long-case final evaluation independently reloads its
endpoint checkpoint. **Improved chemical quality is not a smoke acceptance rule**.
Long cases restart the original authenticated checkpoint, not a smoke-trained
decoder. The unchanged reference is preserved and reevaluated with the same
new evaluator rather than copied from a differently sampled historical report.

The implemented orchestration is
[decoder_finetune.py](../src/latentmol/decoder_finetune.py), with reusable
[decoder_regularization.py](../src/latentmol/decoder_regularization.py),
[decoder_quality_evaluation.py](../src/latentmol/decoder_quality_evaluation.py)
and [decoder_validity_study.json](../configs/decoder_validity_study.json).
Epoch validation uses 256 protected validation graphs and the independent
256-point prior bank; final reconstruction uses all 7,347 test graphs.

The native study is **one serial `gpuv100` 23-hour allocation**, one exclusive
GPU, four cores and 16 GB host memory per core (64 GB total), released after
CPU engineering checks. Its **79,200-second controller budget** includes
smoke/reference work; each smoke arm has a 1,800-second training budget.
Before starting a long case, the controller requires at least its
**10,800-second training budget plus 1,800 seconds** remaining. Training checks
elapsed time before each batch; this is not a hard timeout around validation
or final evaluation. The reservation therefore does not prove that expensive
evaluation or all six cases fit within the allocation. Retain ordinary failed
and time-censored cases and `not_attempted_time_reservation` cases; the final
summary fails if any case is incomplete. There is no automatic continuation.
Ten requested epochs are not ten completed epochs; unequal actual updates
after censoring are not a matched-update result.

Report both the six-case cohort denominator and the actual completed/stalled/
failed/unattempted states, training and evaluation budgets and any continuation
request. A successful wrapper or engineering smoke is not a completed quality
comparison. CPU fixtures alone do not establish the actual-checkpoint GPU
smoke, full-data fine-tuning or any new decoder quality improvement.

## Runnable HPC stages and source070 evidence

The [submission helper](../scripts/hpc/submit_decoder_validity.py) prints a
verified plan unless `--submit` is supplied. It authenticates the original
checkpoint, dataset/split, config and immutable source, retains unique intent
and job receipts, and refuses duplicate or uncertain submissions. Native
submission checks that the exact matching CPU receipt is `DONE` and declares
`done(CPU_JOB)` as its scheduler dependency. For this release, the commands
have this form inside the authenticated HPC release/environment:

```bash
python scripts/hpc/submit_decoder_validity.py --stage cpu \
  --release /work3/s234818/latentmol/releases/source-070-decoder-validity-attempt001 \
  --archive-sha256 711a929014c5199929f420c253b8176ef9605008ddc765ec6a4203549060eee2 \
  --receipts /work3/s234818/latentmol/submissions/source070-decoder-validity-attempt001 --submit

python scripts/hpc/submit_decoder_validity.py --stage native \
  --release /work3/s234818/latentmol/releases/source-070-decoder-validity-attempt001 \
  --archive-sha256 711a929014c5199929f420c253b8176ef9605008ddc765ec6a4203549060eee2 \
  --receipts /work3/s234818/latentmol/submissions/source070-decoder-validity-attempt001 --submit
```

These examples describe stage invocation, not an additional submission.
[CPU checks](../scripts/hpc/decoder_validity_tests.sh) request queue `hpc`,
two cores, 4 GB/core and 30 minutes; [native study](../scripts/hpc/decoder_validity_study.sh)
runs the smoke followed by the six cases inside one V100 allocation. The
release guard rechecks archive/source identity in each allocation.

Source archive
[source-070-decoder-validity-attempt001.tar.gz](../outputs/source-releases/source-070-decoder-validity-attempt001.tar.gz)
SHA-256 is `711a929014c5199929f420c253b8176ef9605008ddc765ec6a4203549060eee2`;
scientific source manifest is
`b2f16906dcde473f9a029ab6862c9002f2340c46904e840c5dcec5396374f2cc`.
Config SHA-256 is
`1b3e8b9142dd8300f79dc0fb9c7400e21b1b67cda5ba3e36b068129ac50a023c`.
CPU **29622690 DONE** passed 88 engineering checks with one CUDA-only replay
skipped, in 14.23 seconds. Its passing toy continuations at coefficients
0/0.1/1 establish frozen encoder, decoder updates and RNG parity. CPU run is
`/work3/s234818/latentmol/runs/source-070-decoder-validity-attempt001-cpu/20261007T130731Z_decoder-validity-cpu_9599f9ce`;
LSF logs are `logs/source-070-decoder-validity-attempt001-cpu.29622690.{out,err}`.
Native **29622699 submitted/PEND** has campaign output
`/work3/s234818/latentmol/runs/source-070-decoder-validity-attempt001-decoder-study`
and wrapper root `runs/source-070-decoder-validity-attempt001-native`.
Its real-checkpoint smoke and six long cases have no result at this snapshot.
The downloaded [engineering-evidence archive](../outputs/hpc-results/source070-decoder-validity/engineering-evidence.tar.gz)
has SHA-256 `8c5d474393ad65c7b8095f73ef55c7e6913ce655087ba51f60e7bb5d9469689e`;
15 members were extracted and all 14 data-file hashes, including the scheduler
snapshot, verified by the coordinator using local standard-library metadata
checks only. The fifteenth member is the digest manifest
`packed_file_sha256.json`; the outer archive SHA-256 covers all 15. Evidence includes the
[completed CPU run/environment/command](../outputs/hpc-results/source070-decoder-validity/extracted/cpu/run.json),
[CPU pytest stdout](../outputs/hpc-results/source070-decoder-validity/extracted/cpu/command.log),
[native submission receipt](../outputs/hpc-results/source070-decoder-validity/extracted/receipts/native-receipt.json)
and [final CPU DONE/native PEND snapshot](../outputs/hpc-results/source070-decoder-validity/extracted/scheduler_snapshot.json).
CPU ran on `n-62-31-7`, Python 3.11.9, Torch 2.5.1+cu121, RDKit 2024.9.6 and
pytest 8.3.5. SSH closed normally after the final status check; queued work
continues independently.

All numerical fixtures, preprocessing, smoke training, fine-tuning and
evaluation run on DTU HPC. Local work is static review, metadata/file hashing,
packaging and transfers. The sealed archive retains the pre-result documentation
snapshot; later status additions here do not modify its scientific manifest.
Every failed or completed stage also belongs in the [ledger](05_experiment_ledger.md).
