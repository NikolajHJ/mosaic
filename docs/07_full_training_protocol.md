# Full-data AE/VAE training protocol

Adopted protocol: 27 September 2026. Data preparation **29503823 completed** with
the intended 215,000/7,347/7,347 split. Corrected source013 validation **29503855**
passed 77 tests (two skipped) and the actual membership audit. Source013 GPU
launches failed before experiment creation. Source014 jobs **AE 29503879 / VAE
29503880** then passed GPU RNG preflights but failed on their first training batch
because the CUDA loss kernel rejects strict deterministic execution. No optimizer
update or epoch completed in those attempts. Source015 validation **29503897**
passed 83 tests (one optional Boltz constructor check skipped), including actual
GPU AE/VAE training/resumption checks and loss/gradient equivalence. Its repeated
cache audit passed. Latest snapshot at **2026-09-27 23:05:37.975769 UTC / 28
September 01:05:38 local** confirms AE **29503898** and VAE **29503899** RUN at
epochs **77 and 76**, respectively. Both completed 60 epochs and entered the
mandatory lower-rate phase at LR **0.0003**. Test evaluation has not started.
At matched epoch 60 over 7,347 validation molecules, valid/exact counts were
AE **6,578/6,236** and VAE **6,028/5,571**; VAE prior validity was **13/1,024**.
The epoch-80 decision is pending. No training rule or job submission changed
during status retrieval; earlier failed and cancelled attempts remain in the ledger.

The user authorized AE and VAE training with **215,000 training molecules**, the
remaining eligible molecules assigned to validation/test, and substantial
convergence and chemistry diagnostics. All numerical work remains on DTU HPC.
The [experiment ledger](05_experiment_ledger.md) retains every attempt and outcome;
the [earlier results](04_latent_pipeline.md) remain a separate feasibility phase.

## 1. Question and comparison

Determine whether substantially more training data and a longer, staged training
schedule improve raw molecular reconstruction and latent-space quality in the
existing graph AE/VAE architecture. This is a matched AE-versus-VAE comparison
within the new campaign. Its larger dataset, holdouts, training budget and
evaluation coverage differ from the 5,000-molecule pilots, so a direct numerical
comparison with those pilots needs these differences stated.

The AE and VAE share graph schema, preprocessing, data identities, architecture,
reconstruction loss, optimizer settings and diagnostic cohorts. The VAE adds
posterior sampling during training and a KL term. Molecular validity, exact
reconstruction, generation and downstream binding remain separate outcomes.
One seed supports a feasibility conclusion; it does not replace independent
replications. No additional Boltz optimization is implied by this training plan.

## 2. Data identities and protected holdouts

The earlier source audit found 229,694 unique eligible molecules after filtering
249,455 rows, including 18,195 representation round-trip rejections and 1,566
duplicate accepted rows. The completed full-cache preparation repeats those
counts. The observed split is:

| Split | Observed molecules |
|---|---:|
| Training | 215,000 |
| Validation | 7,347 |
| Test | 7,347 |
| Total eligible | 229,694 |

The full-cache dataset SHA-256 is
`560840389f7d85592ef8e16a5566e26b5ab3e5dfa30ebe392366dbf54489974f`;
split SHA-256 is `e15daf547436b66cbbd54ae82f53f153d54e2bd24edee9bf2c5787099cbe6e79`.
The source013 actual cross-cache audit passed in
`20260927T181637Z_full-data-isolation-audit_b1dae78a`: exact counts, zero split
overlaps, and all old 4,000/500/500 identities retained in their corresponding roles.

The preparation seed is 17. The source CSV, canonical ordering, topology-only
schema, stereo removal, rejection policy and duplicate handling remain those of
the earlier prototype. Do not truncate molecules or silently relax the codec to
reach 215,000. If eligible counts differ, record the reason and resolve the split
explicitly before training; do not report the planned counts as observed.

A `reserve_cache` option preserves all 5,000 old memberships: 4,000 old training
identities remain training, 500 validation identities remain validation, and 500
test identities remain test. This prevents earlier holdouts from entering new
training and earlier training molecules from entering the new holdouts. Verify reservation
after canonicalization/deduplication. Record the reserve-cache path/hash, requested
and retained identities, any missing/conflicting identities, and resulting overlap
counts. The new cache needs its own immutable dataset/split hashes; it must not
overwrite the 5,000-molecule pilot cache.

The new manifest must report split disjointness and old/new split overlaps, not
just the requested counts. Reservation preserves identities; it is not a scaffold
split or a claim that all held-out molecules were unseen throughout prior method
development. Both model configurations and continuation/selection rules are fixed
before launch. Each model opens test only after its own continuation decisions
and checkpoint choices are sealed. Independent jobs may finish at different times;
early test metrics must not alter either model's training, selection or rules.
Compare campaign test outcomes after both complete.

The data gate requires the HPC preparation manifest, exact counts/hash checks,
rejection/duplicate counts, reservation audit, and regression checks before GPU
training. Raw-source identity remains the previously recorded CSV SHA-256 unless
a different acquisition is explicitly recorded.

The submitted preparation targets
`/work3/s234818/latentmol/data/zinc215k-seed17-v1`. Its planned manifest audit
includes heavy-atom histograms/quantiles by split and element, charge and bond
counts. CPU job 29503823 passed 30 selected chemistry/data tests in 5.65 seconds
(run `20260927T175935Z_tests-full-data_426fafa6`), then started preparation run
`20260927T175942Z_prepare-zinc215k_270e6965`, now completed. The whole data job took
12 minutes 39 seconds. The partial data-only release does not validate
the in-progress training/resumption code.

## 3. Shared configuration

| Setting | Adopted value / state |
|---|---|
| Training seed | 17 for both models |
| Graph capacity | 50 heavy-atom slots |
| Elements | `NO_ATOM, C, N, O, F, P, S, Cl, Br, I` |
| Charges | Categorical `-2, -1, 0, 1, 2` |
| Bonds | Symmetric `NONE, SINGLE, DOUBLE, TRIPLE, AROMATIC` |
| Encoder/decoder | Same ordered graph encoder and slot/pair decoder as pilots |
| Hidden / latent dimensions | 128 / 128 |
| Message layers | 3 |
| Batch size | 64 |
| Initial learning rate | 0.001 |
| Reconstruction loss | Baseline shared loss; `no_bond_weight=0.1` |
| Other baseline loss weights | Padding 0.2; charge 1.0; bond 2.0; size 0.1 |
| AE latent / KL | Deterministic mean, no KL contribution |
| VAE latent / KL | Posterior sampling in training; standard-normal prior; beta 0.001 |
| VAE KL schedule | Linear warmup over the first 20 global epochs |
| Initialization policy | Same seeded model initialization, recorded as `seeded_initial_model_sha256`; initial full-data runs start fresh |
| Optimizer details | AdamW, weight decay 0.00001, gradient clipping norm 5; shared absolute-epoch LR schedule |
| Checkpoint archives | Every 10 global epochs, plus best reconstruction, best validity and last |
| Planned maximum | 120 epochs per model; conditional extensions after epoch 80 |

The baseline loss is restored for both models; the earlier no-bond-weight-1.0
ablation did not consistently improve validity. The decision to keep 0.1 is not
a claim that it is optimal. Resolved configuration files and hashes, rather than
this summary table alone, determine the executed experiment.

The reconstruction size term supervises the training molecule's atom count. It
is not design-time size control. The decoder can predict `NO_ATOM`, while the
existing Boltz adapter still fixes seed occupancy and has no atom birth/death.

## 4. Staged budget and continuation decisions

| Stage | Global epochs | Learning rate | Decision status |
|---|---:|---:|---|
| Initial matched training | 1–60 | 0.001 | Both models run the shared initial budget |
| Mandatory lower-rate probe | 61–80 | 0.0003 | Both models continue; plateau at 60 alone does not end training |
| First possible extension | 81–100 | 0.0001 | Only if the declared validation trend criterion supports continuation |
| Second possible extension | 101–120 | 0.00003 | Only if the declared validation trend criterion supports continuation |

This protocol provides at least 80 planned epochs per model, with up to 120.
Numerical failure or a diagnosed correctness problem is a failed/interrupted
attempt, not successful early stopping. Scheduling boundaries may divide a stage
into several jobs; global epoch/update counts must not reset on resumption.

Continuation is **model-specific**, using validation only. At epoch 80 and 100,
compare the recent 10 epochs with the preceding 10: epochs 71–80 versus 61–70,
then 91–100 versus 81–90. Source inspection confirms arithmetic means for
each window. Extend that
model by 20 epochs if **any** of the following is met:

- Full-validation reconstruction loss improves by at least **0.5% relative**.
- The fixed 1,024-molecule validation chemistry cohort's valid fraction improves
  by at least **0.01 absolute**, meaning one percentage point.
- The same cohort's exact fraction improves by at least **0.005 absolute**,
  meaning half a percentage point.

For positive prior-window reconstruction mean `L_old`, the relative improvement
is `(L_old - L_new) / L_old`; lower loss is better. Chemistry comparisons use
`fraction_new - fraction_old`, with all 1,024 attempts in each denominator.
Best-validity selection maximizes fixed-probe validity, then exact fraction,
then prefers lower full-validation reconstruction loss. Complete ties retain
the earlier epoch. Test metrics never enter continuation or selection.

If no criterion is met, stop that model and report **plateau under the current
schedule**, not proof of convergence. A model still improving at the 120-epoch
cap is **budget-limited**. Preserve the exact window metrics and decision, including
small changes that miss a threshold; do not round metrics before applying it.

At epochs 60, 80, and any later stage boundary, record the checkpoint identities,
validation trend evidence, selected action and reason. The mandatory 61–80 probe
tests whether a lower learning rate still improves the model. Reaching the cap,
meeting a prespecified plateau rule and failing numerically are distinct outcomes.
Do not call a run converged merely because its scheduler job ended.

Retain exact global epochs, optimizer updates, elapsed time and GPU/device details.
The epoch-60 checkpoints are the declared equal-budget comparison, and both models
also receive the mandatory stage through 80. If later budgets differ, report the
matched epoch-60 result and each model's selected final checkpoints and actual
compute separately; final endpoints must not be presented as equal-compute.

## 5. Diagnostics and selection

Confirmed diagnostic coverage:

| Diagnostic | Cohort / cadence | Purpose |
|---|---|---|
| Training reconstruction probe | Fixed seeded 2,048 training molecules, deterministic latent mean | Compare train/validation reconstruction without VAE sampling noise |
| Validation reconstruction probe | Fixed seeded 2,048 validation molecules, deterministic latent mean | Matched cohort trend and generalization gap |
| Full validation reconstruction | All 7,347 validation molecules each epoch if retention matches | Selection/trend evidence beyond a small prefix |
| Per-epoch chemistry | Fixed seeded 1,024 validation molecules | Raw validity, exactness and failure trends with stable denominators |
| Full validation chemistry | Every 10 global epochs and phase boundary | Confirm aggregate chemistry beyond the fixed 1,024-molecule cohort |
| Stage-end VAE prior draws | 1,024 standard-normal draws | Generation assessed separately from reconstruction |
| Stage-end local neighborhoods | 256 seeded validation molecules | Baseline-relative valid changes, rescues and size/identity stability |

Use deterministic means for comparable reconstruction diagnostics. The VAE's
stochastic training objective is not the same quantity as its deterministic-mean
reconstruction probe; plot and label them separately. Keep the same cohort
identities and perturbation/draw seeds across AE/VAE wherever the assumptions
match. An AE has no trained standard-normal prior and must not be presented as
though it does.

Retain total/component reconstruction losses, KL/beta for VAE, learning rate,
epoch/update counts and finite-loss/gradient diagnostics. Chemical reporting must
include raw valid/attempted and exact/attempted counts, atom/charge accuracy,
bond precision/recall, size error, and first-failure categories. Invalid molecules
remain in the attempt denominator; never silently repair outputs or report only
successful decodes. First-failure categories are not independent counts of every
chemical defect.

For convergence assessment, compare deterministic train/validation probes,
full-validation losses and fixed-cohort chemistry over time. Stage-end latent
statistics should distinguish active mean dimensions, posterior variance and KL
from simple reconstruction performance. Report prior validity and uniqueness with
their denominators, and neighborhood changes relative to each seed's unperturbed
decode. Available size-bin/chemistry-subgroup audits and example outputs should
retain cohort identities and counts; added diagnostics are not considered run
until their artifacts are observed.

Selection uses validation only. Save and report best-reconstruction,
best-validity and last checkpoints; each criterion's exact metric/cohort and
tie-breaking rule must be recorded. A loss minimum and chemistry maximum may
occur at different epochs. `best.pt` minimizes full-validation reconstruction;
`best_validity.pt` uses the fixed-probe lexicographic comparator defined above.
Every stage evaluates its last checkpoint on validation. Final selection covers
the common epoch-60 checkpoint, best reconstruction, best validity and last,
deduplicating identical underlying checkpoints while retaining their roles.
Full validation and final evaluation both use batch size 64 and the same seeded
full-split order so batch-normalized weighted losses are comparable.

After a model's continuation/selection decisions are frozen, evaluate its sealed
test split for the declared checkpoints and record the full protocol and results.
Do not reuse that test evaluation to request more training, choose a different
loss or silently relabel a checkpoint, for that model or the other model still
running. Configurations/rules are fixed before either launch; compare test results
after both finish. A later study needs an explicitly separate evaluation plan.

## 6. Resumption and durable evidence

Persistent optimizer/RNG resumption is implemented in the new trainer and passed
static review; it was not a capability of the historical source010 trainer.
Format-1 checkpoints remain compatible with evaluation and add versioned
continuation state: AdamW, global epoch/update, absolute LR and KL schedule position,
selection trackers, Python/NumPy/Torch/CUDA RNGs and independent loader generators.
NumPy RNG keys are serialized as tensors/plain data for safe checkpoint loading.
Source/runtime/data/split/schema/config/cohort checks reject incompatible resumes.
Only the configured epoch horizon may extend; changing phase stop boundaries
does not change the underlying 120-epoch configuration.

Continuation is supported at completed epoch boundaries. The workflow resumes
between stages automatically in the same process, retaining the exclusive GPU
context. A scheduler interruption requires explicit manual recovery through the
trainer and its saved checkpoint; restarting the whole workflow is not automatic
recovery and would start a new workflow. Do not claim transparent mid-epoch or
whole-workflow recovery. New exact interrupted-versus-uninterrupted tests cover
zero and one data-loader worker and passed within the source013 CPU suite.
Source014's separate GPU RNG checks passed, but its first training batch exposed
a CUDA deterministic-kernel limitation. Actual GPU training/resumption checks
passed for the source015 correction: AE and VAE with zero and one loader worker.
RNG-only checks are distinguished from those actual training/resumption tests.

Immutable archives every 10 epochs preserve earlier states. Best-reconstruction,
best-validity and last checkpoints need their own identifiers/hashes and selected
epoch/metric. A resumed process receives a new run/attempt identity linked to its
parent checkpoint; every scheduler job belongs in the ledger. No job ID is a
success claim.

Required provenance includes executed source archive SHA-256, Git base/dirty
state as available, resolved model/loss/training config hashes, raw/prepared/split
and reserve-cache hashes, probe indices/seeds, environment/actual GPU, checkpoint
hashes, learning-rate/beta schedules, complete epoch history, stage decisions,
evaluation manifests and errors. Retain unsuccessful attempts and partial
artifacts with their validity clearly labeled.

## 7. Pending decisions and execution register

| Item | State |
|---|---|
| Full eligible-data counts and immutable cache hashes | Verified 215,000/7,347/7,347; hashes above; actual reservation/disjointness audit passed |
| Continuation criterion | Confirmed model-specific arithmetic-window comparison: >=0.5% relative reconstruction gain OR +1pp valid OR +0.5pp exact |
| Initialization / optimizer / selection | Confirmed same seed with recorded initial-state hash; AdamW decay 1e-5, clipping norm 5; lexicographic validity/exact/loss comparator, earlier full ties |
| Resumption/diagnostic implementation and regression checks | Source015: 83 passed, one optional skip; actual GPU resume and loss/gradient equivalence passed; actual audit passed |
| Data-only source archive | `source011-data`; SHA-256 `22729bccf6b72e0b412e8fdbd09fa43beb08e99e2d3a9d6db234c3dddb1c9d7a` |
| Full source archive | `source012`; SHA-256 `37bb498a6e72690dddd3f104384df92ed2139a6cc1ebee88ba875b08c3c787a2` |
| Corrected full source archive | `source013-full`; SHA-256 `9641779261bf1bd125ae51e34535ea92754858cf90bd7aedcb3bfc35f47110ac` |
| Direct-launch source archive | `source014`; SHA-256 `03e7ee0c710ac604e2322d5ef39790715b3ae6185115d92695acd3ab02b3eb45` |
| CUDA-loss correction archive | `source015`; SHA-256 `375c7f6a5a67f229d5c95d307413047b26e244de82a1c71ff09a0471bc192f12` |
| Scheduler IDs and resources | Validation 29503897 DONE; AE 29503898 / VAE 29503899 RUN at the 23:05:38 UTC snapshot on separate A10 allocations; four cores, 4 GB per core, 900-minute limit each |
| Full-data AE/VAE progress | AE epoch 77 / VAE epoch 76, both LR 0.0003; latest fixed-cohort valid/exact AE 990/968 and VAE 949/914 out of 1,024; test unopened |

The corrected full regression job used four CPU cores, 4 GB per core and a
60-minute limit. Matplotlib 3.10.1 installed during source012; source013 preserved
its pin and passed the tests and full-cache audit. GPU launch failures and the
later deterministic-loss error are retained as separate incidents. A10 is the
selected GPU for both models; record actual device/runtime. Source015 passed its
actual CUDA training/resumption gate before the current full jobs started.

The coordinator will update these fields with evidence. The earlier 5,000-molecule
pilots and successful fixed-context Boltz audit remain historical results; they
must not be presented as full-data training evidence.

The running AE and VAE manifests both confirm 215,000 training molecules, 7,347
validation molecules and 2,048-molecule probes. Their common initial model-state
SHA-256 is `2a553d529be5e217a2d804751d09979113b729395a4c6765d5ac46890cb0e6ed`.
This verifies matched initialization; it does not imply identical subsequent
training trajectories. Both subsequently completed epoch 1 and saved checkpoints.

Both subsequently completed the initial 60-epoch phase and resumed within their
existing workflows as `20260927T220339Z_ae-train_406d4e54` and
`20260927T220435Z_vae-train_5045659f`. Latest losses are 0.0127113 and 0.0113253;
they are higher than the respective epoch-61 values despite better chemistry
counts. Neither monotonic improvement nor convergence is established. Apply the
original epoch-80 continuation rule without adjusting it to these observations.
