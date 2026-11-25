# Stage 3: bounded schema-matched denoiser pilot

Updated 28 September 2026. Immutable **source022** CPU preflight **29505906 is
DONE**: **three tests passed in 4.26 seconds**, in completed run
`20260928T092041Z_tests-stage3-denoiser-preflight_a3a08d56`.
GPU pilot **29505909 is DONE**, with three passed tests and completed run
`20260928T092202Z_stage3-schema-denoiser-pilot_ac19a401`. It used `gpua10`,
four CPU slots, 8 GB per slot, one A10 GPU in exclusive-process mode and a
two-hour limit. The workflow completed, but **feasibility failed**: selected
denoiser outputs were valid in **0/256** attempts, and the local gradient gate
returned `passed=false`. **The denoiser is not promoted to guided optimization.**
All tests and numerical work run on DTU HPC. This is a small denoising
autoencoder using the project's graph schema, **not a DeFoG checkpoint**, a
molecular likelihood, a validated chemical score or a Boltz combination.

Local source archive:
`outputs/source-releases/latentmol-source-022-denoiser-pilot.tar.gz`.
HPC archive:
`/work3/s234818/latentmol/releases/source-022-denoiser-pilot.tar.gz`, with the
extracted `source-022-denoiser-pilot` release beside it. Source SHA256:
`363c4d9f0303bff8e775d404adab5f74d01a4fe5533f7b710b0b662508110178`.
The source and failed scientific result remain immutable; a completed scheduler
job does not override the saved evaluation and gate outcomes.

## Observed outcome: completed workflow, failed feasibility

The four-epoch pilot selected **epoch 4** by development-validation reconstruction
loss. Both models received the same corrupted Stage 1 pilot cohort.

| Model on fixed corrupted validation | Reconstruction loss | Valid hard outputs | Exact reconstructions |
|---|---:|---:|---:|
| Original frozen AE | 52.624661 | 4 / 256 | 0 / 256 |
| Selected denoiser, epoch 4 | 1.785820 | 0 / 256 | 0 / 256 |

| Epoch | Training loss | Development-validation loss | Valid denoiser outputs |
|---|---:|---:|---:|
| 1 | 20.811393 | 10.467269 | 0 / 256 |
| 2 | 5.962792 | 3.546359 | 0 / 256 |
| 3 | 2.507536 | 2.098083 | 0 / 256 |
| 4 | 1.827836 | 1.785820 | 0 / 256 |

The loss decreased throughout this bounded run, but lower weighted reconstruction
loss did not yield a valid hard molecule. That loss averages local atom, charge,
bond and size errors; hard chemical validity depends on a jointly consistent
discrete graph. Improving the former does not guarantee the latter. Four epochs
with falling loss do not establish convergence or rule out a different training
protocol. They also do not justify promoting this checkpoint.

The fixed-reference soft consistency value was **0.0665876642**. At selected
latent coordinate 91, the analytic derivative was **-1.1588877e-10** and the
central difference was **0.0**. Absolute error was within the numerical tolerance,
but total latent-gradient norm was only **4.3467543e-10**, below the required
`1e-8`. Consequently **`passed=false`**. This supplies no useful-gradient evidence
at the tested point. The cause of the near-zero signal is not established by
this one-coordinate check; further diagnostics must separate decoder/denoiser
behavior and the choice of scalar rather than infer a general impossibility.

Selected checkpoint SHA256:
`bf68ed7d6cd7da9450fe1eed8a8c1b8c0fe32693a7e93fc29ea9c283efc1845a`.
The complete downloaded result archive is
`outputs/hpc-results/stage3/denoiser-pilot-source022-results-20260928.tar.gz`,
with verified local/HPC SHA256
`12618cc77436c635ca75a9cfecab09c1cdca318f0ceda42cf7b4e262b5669004`.
Three run directories were extracted. The pilot's `summary.json`, `history.json`,
`gradient_gate.json`, checkpoint and manifest are under
`outputs/hpc-results/stage3/20260928T092202Z_stage3-schema-denoiser-pilot_ac19a401/`.
The local archive hash, summary and epoch history were checked for this report.

The source023 post-hoc failure diagnostic completed, as recorded below.
No retraining run or Boltz combination is submitted. The source022 outcome
remains failed feasibility; the remaining methods sections describe that run.

## Source023 post-hoc failure diagnostic — completed

Immutable **source023** CPU job **29505949 is DONE**. Run
`20260928T092654Z_stage3-denoiser-failure-audit_3b2a9bed` records `completed`.
The allocation was `hpc`, four CPU slots, 4 GB per slot and a one-hour limit.
Local archive:
`outputs/source-releases/latentmol-source-023-denoiser-failure-audit.tar.gz`.
HPC archive:
`/work3/s234818/latentmol/releases/source-023-denoiser-failure-audit.tar.gz`,
with the extracted sibling release. Source SHA256:
`122b22a3c7a4d8a63f63c9795dba62d0a531817d799130a1db9cbd6496c9c8af`.

Its
`src/latentmol/denoiser_diagnostics.py` and
`scripts/hpc/stage3_denoiser_failure_audit.sh` loaded the immutable source022
selected denoiser and original AE, verified their recorded checkpoint hashes,
dataset/split identities, full validation-index hash/range and graph schema,
and froze both. It inspected the **first 128 members of the
existing Stage 1 pilot validation cohort**, without retraining, choosing a new
checkpoint or accessing the promotion reserve/test set.

Two paired conditions were evaluated: clean graphs at time 0, and categorical
corruption at time 0.25 with a fixed seed. Both models see identical inputs
within each condition. The audit reports hard validity/exactness and failure
reasons, hard-size histograms, mean expected size, reconstruction/atom/bond
losses, accuracy at clean occupied atom slots, and recall of true bonded pairs.
Bonded-pair recall measures presence of an edge, not correctness of its bond
order. Up to eight examples per model/condition retain target, decoded identity,
failure, hard size and edge count; examples are the first attempts, not a
representative random sample of failures.

The audit uses CPU evaluation. Reusing seed 423 does **not** reproduce CUDA
corruption draws from source022, so fixed noise means reproducible paired inputs
within this diagnostic, not an exact replay of the pilot. Time 0 is also outside
the denoiser's training range [0.10, 0.35]; the clean condition can expose a
problem but cannot alone distinguish time extrapolation from loss of clean
reconstruction ability. This audit does not test or repair the latent-gradient
failure.

The same 128 clean targets contain **3,269 true undirected bonded pairs**.
Counts below aggregate over all 128 attempts in each condition; edge recall
ignores bond order.

| Condition / model | Valid / exact | Predicted bonded pairs | Active-atom accuracy | Untyped edge recall |
|---|---:|---:|---:|---:|
| Corruption 0.25 / frozen AE | 0 / 0 | 2,707 | 0.7032 | 0.6115 |
| Corruption 0.25 / denoiser | 0 / 0 | 5,437 | 0.8317 | 0.8180 |
| Clean, time 0 / frozen AE | 125 / 124 | 3,271 | 1.0000 | 0.9997 |
| Clean, time 0 / denoiser | 0 / 0 | 6,253 | 0.9382 | 0.9385 |

Every denoiser output failed sanitization in both conditions (**128/128** each).
The corrupted frozen AE had 120 sanitization failures and eight disconnected
outputs; the clean frozen AE had two sanitization failures and one disconnected
output. On the in-range corrupted condition, denoiser reconstruction loss was
**1.812483**, versus **53.165495** for the frozen AE. Better local loss, atom
accuracy and edge recall coexist with excess predicted bonds and zero valid
hard decodes. Edge recall alone rewards recovering bonds without penalizing
additional false bonds, and the aggregate counts do not establish that
overbonding is the sole cause of every failure.

The clean/time-zero stress test is consistent with poor retention of clean
behavior, but its out-of-training-range time prevents a clean attribution to
backbone forgetting alone. CPU corruption is newly drawn, so the corrupted
frozen-AE result here does not contradict source022's 4/256 valid CUDA result.
**There is no denoiser promotion.** A bounded loss/clean-retention repair ablation
is being prepared; it is not yet a submitted experiment or a DeFoG/Boltz
combination.

The downloaded result archive is
`outputs/hpc-results/stage3/denoiser-failure-source023-results-20260928.tar.gz`,
with matching local/HPC SHA256
`4a81673335d1ca1267b10dd61ddbf8efbb4314900fc321ba547e4f145f97a91d`.
The extracted run is under
`outputs/hpc-results/stage3/20260928T092654Z_stage3-denoiser-failure-audit_3b2a9bed/`.
Its summary and local archive hash were checked for this report.

## Question and starting point

The previous training-support diagnostic described selected structures and
training neighbors; it did not supply a differentiable chemical prior. This
pilot asks two narrower questions: can an AE initialized from the existing
model repair specified corruptions better after a short fine-tuning run, and
can a declared soft-input scalar propagate a locally checked gradient to the
original AE latent variable?

The starting backbone is the validation-selected AE at epoch 108:
`runs/20260928T002738Z_ae-train_2f61b913/best_validity.pt`, SHA256
`1b1d2a32aedfadb376d5a5e2d8802d574ed5c1d72851ed79517acaef418e9b95`.
Its original training used 215,000 molecules. The pilot's 5,000-molecule budget
is **additional denoising fine-tuning**, not the model's entire training history.
An independent frozen copy supplies the paired control and later latent decoder.

## Implementation boundaries

| File | Responsibility |
|---|---|
| `src/latentmol/denoiser.py` | Noise-conditioned AE wrapper and categorical corruption |
| `src/latentmol/denoiser_study.py` | Seeded subsets, fine-tuning, paired evaluation, checkpoint selection and derivative diagnostic |
| `tests/test_denoiser.py` | Corruption and differentiable-input regression contracts |
| `scripts/hpc/stage3_denoiser_pilot.sh` | Require an LSF allocation, verify the starting checkpoint and frozen Stage 1 cohort hashes, run regressions and launch the GPU pilot |

`SchemaDenoiser` copies the original graph encoder/decoder weights. It encodes
the graph deterministically to a mean vector, adds a learned time embedding
and decodes. The time embedding is a scalar-to-latent MLP with a SiLU
nonlinearity; its final layer starts at zero. The backbone and time embedding
are trainable during this pilot. This keeps the ordered 50-slot graph schema,
128-dimensional latent/hidden architecture, element/NO_ATOM vocabulary, formal
charges and five bond classes already used by the AE.

The slot ordering remains the dataset's canonical ordering. This experiment
does not establish permutation-invariant denoising or reconstruction under a
different ordering. Nor does its handling of NO_ATOM implement variable atom
count in the current fixed-occupancy Boltz adapter.

## Data, corruption and training

Use the existing `zinc215k-seed17-v1` cache. The dataset reader verifies the
NPZ hash, schema and actual split digest. The initialization must be an AE
checkpoint with the same dataset identity. The study records data/split/schema,
checkpoint hash/epoch, explicit chosen indices and index hashes.

| Setting | Initial bounded value |
|---|---:|
| Master seed | 17 |
| Fine-tuning molecules, TRAIN only | 5,000 |
| Development-validation molecules | 256 |
| Epochs / batch size | 4 / 64 |
| AdamW learning rate / weight decay | 0.0002 / 0.00001 |
| Gradient clipping norm | 5 |
| Training replacement probability | Uniform per graph in [0.10, 0.35] |
| Fixed validation replacement probability | 0.25 |

Training indices use a fixed seed offset. Validation deliberately reuses the
**256-member Stage 1 AE pilot development cohort**, from
`runs/20260928T063833Z_ae-latent-stage1_580abd75/cohorts.json`, whose recorded
SHA256 is `dd0645a76ce6074398e3d196e999116f11b134d8abbebf9726368ebc2c94d3e2`.
The shell verifies this cohort hash before launch. The reader requires the
expected count, unique in-range indices and no overlap
with that file's **200 reserved promotion indices**. It records the actual
cohort-file hash and selected-index hash in the run manifest. Reusing this
development cohort preserves the unused reserve; it is not independent new
validation evidence. No test-split evaluation occurs here.

Training order and corruption have separate seeded generators. Validation
resets its noise generator on each evaluation, so both models and all epochs
see the same corrupted inputs.

For each clean graph, eligible slots comprise its original occupied slots plus
the next four padding slots, capped at the graph capacity. Independently chosen
atoms and charges are replaced with uniform categorical draws at the sampled
probability. Bond corruption draws each eligible unordered pair once and
mirrors it; the diagonal remains NONE. After atom corruption, bonds with either
endpoint absent are set to NONE and charges at absent atoms are neutralized.
This preserves the shared hard/relaxed occupancy contract, with assertions
added to the corruption regression. That correction addresses the initial
static-review concern and is covered by the passed CPU preflight; the GPU
allocation reruns the regressions before training.
A replacement draw can reproduce the original category, so the replacement
probability is not the fraction of labels that actually change. Corruption may
create chemically invalid graphs, which are deliberate repair inputs.

Training targets are the original clean ordered graphs. The existing
reconstruction loss uses padding weight 0.2, no-bond weight 0.1, charge weight 1,
bond weight 2 and normalized size-loss weight 0.1. No KL or Boltz objective is
used. Charge and bond losses use clean occupied slots/pairs. Reported epoch
losses aggregate batch losses weighted by batch size; their class weighting
still follows the existing reconstruction-loss implementation.

## Paired hard-graph evaluation and selection

Each epoch evaluates the fine-tuned denoiser and original frozen AE on identical
fixed-corruption inputs. The denoiser receives the declared corruption time;
the original AE does not have a time-conditioning module. Both outputs are
hardened using the same argmax, symmetry and occupancy rules, then checked by
the shared chemistry decoder. Count all 256 attempts, including invalid
outputs. Exact reconstruction compares the decoded identity with the original
clean reference SMILES. No post-hoc molecule repair is part of these counts.

Record reconstruction loss, hard validity and exact reconstruction for both
the denoiser and frozen AE. Select `best.pt` by the smallest
development-validation reconstruction loss across the four epochs, preserving
the earlier epoch on an exact tie. This is not selection by chemical validity.

The corruption is uniform categorical replacement, whereas later latent
optimization creates structured soft decoder outputs. Repair performance on
these corruptions does not establish generalization to that latter distribution,
rare structural motifs, synthesis feasibility or out-of-distribution chemistry.
There is no clean-input retention assessment or selected-candidate challenge-set
evaluation in this initial pilot.

## Local latent derivative diagnostic

After selection, freeze the denoiser and original AE. Take the first selected
validation molecule, encode it with the original AE to `z0`, and freeze the
original decoder probabilities `D(z0)` as a reference. The tested scalar is the
sum of mean squared differences between denoiser output probabilities at
`D(z)` and that **fixed** reference, separately for atoms, charges and bonds.
Time is fixed at 0.25; no categorical sample is drawn inside this derivative
path. The path is `z -> frozen AE decoder -> frozen denoiser -> scalar`.

This is a fixed-reference consistency scalar, not an unconditional realism
score or the discrepancy between a graph and its own moving denoised output.
The local check chooses the latent coordinate with the largest absolute
analytic derivative, uses one central-difference epsilon of 0.01, and compares
absolute error with `max(1e-5, 0.25 * abs(finite_difference))`. It also requires
finite analytic derivatives and gradient norm above `1e-8`.

That is a bounded engineering check, weaker than a multi-epsilon convergence
test. Passing does not show a chemically helpful direction. The current
workflow records the gate's boolean outcome without raising an exception for
`passed=false`; a completed scheduler job must therefore not be reported as a
passed derivative gate unless the saved gate explicitly passes.

## Artifacts and decisions

Expected artifacts are `selected_indices.json`, `history.json`, `best.pt`,
`gradient_gate.json`, `summary.json`, metric records and the experiment manifest.
The denoiser checkpoint declares `format_version=1` and saves its weights,
selected epoch, model configuration, loss configuration, graph specification
and run configuration. These are denoiser checkpoint fields; the artifact is
not asserted to be interchangeable with an ordinary AE training checkpoint.
The summary names the selected epoch and paired metrics, retains the local
gate and checkpoint SHA256, and states the claim limit. Record the immutable source, allocation,
tests, run status and transfer hashes in the experiment ledger when supplied.

Assess repair and derivative outcomes separately. Neither alone authorizes a
claim that this model prevents optimization collapse. Any next guidance study
needs a declared useful scalar, stronger derivative checks where needed,
decoder-like perturbation tests and all-attempt hard-molecule comparisons
against the original baseline and trust-region control. No such follow-up or
Boltz combination is recorded as submitted here.
