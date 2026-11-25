# Stage 2: fixed-context Boltz latent pilot

Recorded 28 September 2026 from immutable source017. **Both Boltz pilots
completed successfully**, with 19 selected tests passed per job, passed binary
finite-difference gates and unchanged model/context digests. AE job **29505584**
and VAE job **29505585** each requested one gpua10 GPU, four CPU cores,
4 GB per core and six hours. This is a small engineering screen on one saved
protein–ligand context, not an independent binding or design evaluation.

Source archive SHA256:
`06ef6636810f195b10ff0eea98c4458a297bf3c19685ded3532f96fa578b9686`.
HPC release: `/work3/s234818/latentmol/releases/source-017-stage2`, with the
archive at the sibling `.tar.gz` path. The local archive is
`outputs/source-releases/latentmol-source-017-stage2.tar.gz`.
Submission and later outcomes are recorded in [the ledger](05_experiment_ledger.md).

## Observed pilot outcomes

The coordinator verified completed HPC manifests for AE run
`20260928T073635Z_ae-boltz-latent-pilot_6889d56d` (**2,807 seconds**) and VAE run
`20260928T073724Z_vae-boltz-latent-pilot_a8e9e0e3` (**2,744 seconds**).
All 64 final endpoints per model were retained. Actual completed outer scorer
calls were **672 forwards / 325 backwards for AE** and **633 / 325 for VAE**.
The forward totals include valid hard projections; invalid projections retain
null scores and do not consume a corresponding scorer forward.

The strict engineering outcome below requires a valid hard endpoint different
from both its valid perturbed start and encoded baseline, with fixed-context
binary probability improving by more than 0.0001 over both comparators.

| Model | Target successes / 16 | Trust / 16 | Joint / 16 | Random / 16 |
|---|---:|---:|---:|---:|
| AE | 13 | 0 | 14 | 0 |
| VAE | 5 | 0 | 5 | 0 |

Each model's target and joint arms had **two unique changed hard identities per
arm**. Success counts above include repeated arrivals at the same molecules;
they are not counts of independent molecular discoveries. These results show
that the frozen network can guide hard changes under its fixed-context proxy.
They do not establish a meaningful joint-versus-target advantage, a general
AE/VAE winner, or improved binding.

All scored candidates retained **13 occupied heavy-atom slots**, one saved pose
and the same protein/ligand context. Passing local derivative and frozen-state
checks does not resolve that scoring limitation. In the subsequent source018
native follow-up, **AE 29505758 completed all eight predictions** and **VAE
29505759 is running**. One of two selected AE target endpoints, and zero of two
selected joint endpoints, passed the native binary criterion against both
controls on both seeds. The passing target candidate's continuous affinity
output moved in the unfavorable direction, and its chemistry has not been
characterized. The [native report](13_native_rescore.md) records this limited
predicted-score result; it is not an affinity or binding improvement claim.

Complete local evidence is available in the
[AE run](<C:/Users/nikol/Documents/Codex/2026-09-26/okay-so-i-want-you-to/outputs/hpc-results/stage2/20260928T073635Z_ae-boltz-latent-pilot_6889d56d/summary.json>) and
[VAE run](<C:/Users/nikol/Documents/Codex/2026-09-26/okay-so-i-want-you-to/outputs/hpc-results/stage2/20260928T073724Z_vae-boltz-latent-pilot_a8e9e0e3/summary.json>) directories,
including **65 tensor files per model**, all trajectories/final attempts,
cost and derivative reports, and manifests. Transfer archive
`outputs/hpc-results/stage2/stage2-source017-results-20260928.tar.gz` has matching
HPC/local SHA256
`95d76c36b9f3881db4428fbfb5f608004ee03b39359e2f5ff81df256c2fac65b`.
The archive is approximately 3.8 MB compressed; each original run is approximately
41 MB. This receipt verifies retained artifacts, not a new experiment.

## Why these arms

Stage 1's PCA/KDE restraint failed its promotion criterion for both models.
This pilot therefore compares a simple trust penalty with target-only guidance;
it does not promote KDE as a successful chemical prior. The saved training-code
reference is reused for its mean and per-dimension scale, so each model's
displacements have its own standardized coordinates. KDE density is not an
objective here.

The selected checkpoints remain the sealed best-validation-validity AE epoch
**108** and VAE epoch **94**, described in [the full-training audit](10_full_training_audit.md).

| Model | Checkpoint SHA256 | Source016 reference run |
|---|---|---|
| AE | `1b1d2a32aedfadb376d5a5e2d8802d574ed5c1d72851ed79517acaef418e9b95` | `20260928T063833Z_ae-latent-stage1_580abd75` |
| VAE | `80b22b9c5a5e9010457b49246f830f0425c5578458eab28ddf2a10f5f632cee9` | `20260928T064117Z_vae-latent-stage1_bd6a6a7a` |

The shell script verifies the expected checkpoint hash. The pilot also checks
model mode, graph schema and dataset identity, and verifies the reference's
checkpoint/data/split/schema/mode metadata and 215,000 training-code count.
The reference and prepared-context files are hashed into the run configuration.
No training or original test molecules are evaluated in this pilot.

## What is frozen

Both selected graph models and Boltz remain in evaluation mode with weights
frozen. Only a normalized latent vector changes. Use the existing snapshot:

`/work3/s234818/latentmol/boltz-preparation/source006-attempt001/prepared.pt`.

It comes from the previously prepared stereo-free official example. The saved
13-heavy-atom ligand occupancy, explicit slot-to-atom/token maps, protein context,
pose, reference conformer, atom names, masks, crop and MSA context remain fixed.
The adapter conditions decoder logits on that occupancy before bond gating.
Element, charge and bond probabilities can change; atom birth/death, reordering,
new poses and diffusion derivatives are not implemented in this path.

The differentiable wrapper uses the pinned native affinity network with **zero
recycles**, FP32 and TF32 disabled, including activation checkpointing for memory.
Parameter and context digests are compared before/after the complete pilot;
a change fails the run. The earlier native parity checks are historical adapter
evidence, not automatically new checkpoint-specific results.

The optimized objective is **negative binary affinity probability**. The raw
continuous affinity value is logged but is not optimized. Neither output has
been calibrated as experimental binding evidence for these generated molecules.
The decoder's ordinary expected atom count is also logged, but it is diagnostic:
the scored graph always uses the fixed occupancy.

## Regression and binary-gradient gate

Before the pilot, run selected adapter, pilot and latent-reference regressions.
Before any optimization trajectories, test the exact binary objective at a
separate radius-4 latent point, direction seed **18888**. Require a finite,
nonzero analytic gradient. Compare its directional derivative along the
normalized analytic gradient with central differences at
**0.0001, 0.0003, 0.001, 0.003 and 0.01**.

At least **two distinct epsilon values** must agree within **2% relative error**.
All errors are retained in `binary_gradient_gate.json`; gate failure stops the
pilot. This is a local derivative check at one point and does not prove accurate
gradients or meaningful scores across the whole molecular search space.

## Starts, calibration and updates

Encode the saved seed graph to its deterministic latent mean and standardize it
using the matching training reference. Its hard decode supplies the unperturbed
baseline. For each of two repeats, make eight starts: four at radius **2** and
four at radius **4**, using seeds `23000 + 100*repeat + start_index`.
Both models use the same numerical direction seeds, not necessarily the same
molecular edits. All **16 starts per arm are correlated around one seed/context**.

Calibrate the trust weight at four separate radius-4 directions, seeds
**19000–19003**. Take the median ratio of binary-objective gradient norm to
trust-gradient norm and clamp it to **[0.0001, 10000]**. Require finite nonzero
binary gradients. The resulting per-model weight is frozen for all trajectories;
these calibration points are separate directions, not independent molecules.

The trust penalty is `max(0, ||u-anchor|| - 3.5)^2`. Compare:

| Arm | Update objective |
|---|---|
| Target | Negative binary affinity probability |
| Trust | Weighted trust penalty only; affinity is measured |
| Joint | Negative binary probability plus the same weighted trust penalty |
| Random displacement | One random direction with length matched to that start's target-arm endpoint displacement |

Target, trust and joint arms receive **ten normalized gradient steps of length
0.25**, with a maximum displacement **2.5** from the shared perturbed start.
A finite zero gradient leaves the point unchanged. The trust-only radius-2
starts are already inside the trust region and therefore stay stationary;
report radius strata when interpreting unchanged molecules. Gradient norms,
weighted trust norms and pairwise gradient cosines expose scale and conflict.
Normalizing the combined update is distinct from gradient-norm clipping.

Save all 11 states in each updated trajectory, including normalized latents,
fixed-occupancy soft graphs and hard labels. The random arm saves its endpoint.
Score hard projections at trajectory start/end, retaining invalid outputs with
null scores instead of discarding or repairing them.

## Cost accounting

`oracle_cost.json` is updated before and after each outer scorer call and
backward attempt, including failures. Counters are split by purpose. They count
wrapper invocations, not individual internal blocks or activation-checkpoint
recomputations. Tests, queue time and later native re-featurization are separate.

| Component | Forward calls if completed | Backward calls |
|---|---:|---:|
| Shared encoded baseline | 1 relaxed, plus 0–1 valid hard projection | 0 |
| Binary finite-difference gate | 11 | 1 |
| Four weight-calibration points | 4 | 4 |
| Each target trajectory | 11 relaxed, plus 0–2 valid hard projections | 10 |
| Each joint trajectory | 11 relaxed, plus 0–2 valid hard projections | 10 |
| Each trust trajectory | 11 relaxed, plus 0–2 valid hard projections | 0 |
| Each random endpoint | 1 relaxed, plus 0–1 valid hard projection | 0 |

For all 16 starts, the submitted complete protocol entails **560 relaxed
forwards, up to 113 additional hard-projection forwards, and 325 backwards per
model**. These are protocol counts, not observed consumption. Actual attempts,
completed counts and run time must be reported, especially after an early failure.
The arms have different costs; equal start/step counts do not establish equal-
compute efficacy. The random arm matches target displacement, not joint/trust
displacement or an expensive-oracle budget.

## Endpoints and limits of an apparent improvement

Record raw validity, changes from the encoded baseline, and changes from both
the valid perturbed start and valid baseline. The strict fixed-projection metric
also requires binary probability to improve by **more than 0.0001 over both**.
Keep all attempted trajectories in denominators, preserve baseline/start failures,
and report unique changed molecules as well as repeats. If the encoded baseline
is invalid, strict valid-baseline success cannot be claimed.

The hard-projection score uses one-hot molecular features in the **same frozen
context**. It does not rebuild ligand features or generate a new pose. A larger
relaxed probability, or even this stricter projection metric, is only an
engineering signal. Fresh native ligand featurization and repeated new poses
for a predeclared subset from **every arm and its baseline**, at matched budgets
and with failures retained, are required before any predicted-binding improvement
claim. No such fresh-pose stage is included in source017's pilot script.

The pilot cannot establish experimental affinity, selectivity, synthesizability,
variable molecular size or a general AE/VAE winner. Positive/negative controls,
more molecular seeds and independent target contexts remain necessary for later
scientific comparisons under [the broader plan](09_next_experiment_plan.md).

## Artifacts and companion audit

Expected pilot artifacts include `run.json`, `metrics.jsonl`, `baseline.json`,
`baseline_state.pt`, `binary_gradient_gate.json`, `weight_calibration.json`,
`oracle_cost.json`, trajectory/random tensors, `final_attempts.json` and
`summary.json`. Partial final-attempt and cost records are saved during execution.
Run configurations record source, checkpoint, affinity weight, prepared context,
reference and data identities; none is evidence of a completed result by itself.

Source017 also supplied CPU job **29505583**, which has completed its paired
interpolation audit after one test passed. Its result is documented in
[the Stage 1 report](11_stage1_latent_study.md): VAE had higher intermediate
validity on the fixed paired paths, but shared-valid midpoint evidence was too
small to support semantic continuity. This audit uses existing artifacts and
does not count as a Boltz pilot result.
