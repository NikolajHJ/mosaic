# Matched stochastic decoding and native DeFoG transition projection

Updated 28 September 2026. Corrected source025 CPU preflight **29507180 is DONE**:
**21 tests passed in 8.32 seconds**, run
`20260928T122224Z_tests-stochastic-bridge-preflight-corrected_f5958964`.
Phase-1 bridge jobs **AE 29507219 / VAE 29507220 are DONE**. They used `hpc`,
each with two CPU slots, 4 GB per slot and a 30-minute limit, on the same
immutable source025. AE produced **64/64 valid draws, three unique molecules**;
VAE produced **27/64 valid, two unique molecules**. Most draws carried charge.
Phase 1 used saved source017 states; **no DeFoG or Boltz call, new optimization
or variable-size experiment occurred**.

The original preflight **29507173** failed before any tests because its command
referenced nonexistent `tests/test_representation.py`; failed run
`20260928T122056Z_tests-stochastic-bridge-preflight_4ac765f1` is retained. This
was an orchestration error, not a sampler failure. The corrected invocation
used `tests/test_sampling.py` and `tests/test_models.py`.
Phase 2 depends on
native checkpoint loading and a tested wrapper around DeFoG transitions.
Native Boltz rescoring follows only after interface and retention checks.
All numerical work is on DTU HPC; no new latent optimization occurs here.

Local source archive:
`outputs/source-releases/latentmol-source-025-stochastic-bridge.tar.gz`.
HPC archive:
`/work3/s234818/latentmol/releases/source-025-stochastic-bridge.tar.gz`.
The coordinator verified its HPC SHA256:
`82fe0765abb525e97fd832cfe0b93df48b7ddab7ef0064f19d54066c551b150e`.

## Why this comparison

The completed fixed-context pilot produced only a few distinct guided hard
molecules; native standardization erased some changes, and fresh-pose results
did not establish a useful design advantage. Both schema-matched denoiser pilots
then failed their development gates. Those failures do not validate a pretrained
DeFoG projector or show that categorical decoder sampling will help. They motivate
measuring these two operations separately before combining gradients or scaling
Boltz. Earlier outcomes remain in [the ledger](05_experiment_ledger.md).

Stochastic decoding here means drawing graph categories from the saved
**fixed-occupancy graph probabilities** produced by both frozen decoders. It is
not an AE-versus-VAE comparison in which only the VAE receives posterior noise.
The saved states already incorporate each model's previous optimization.
Neither weights nor those states are updated or re-decoded here.
These are correlated starts around one target/ligand context, not an efficacy
estimate across independent molecular seeds or proteins.

## Observed phase-1 results

Completed runs are AE `20260928T122329Z_ae-stochastic-bridge_9ff8ddc9` and VAE
`20260928T122331Z_vae-stochastic-bridge_5a23a72f`. These are 64 attempted draws
per model, preserving invalid outputs and repeated identities.

| Outcome | AE | VAE |
|---|---:|---:|
| Valid hard draws | 64 / 64 | 27 / 64 |
| Unique valid molecules | 3 | 2 |
| Valid draws identical to their parent argmax | 60 / 64 | 16 / 64 |
| Draws with at least one charged atom | 61 / 64 | 56 / 64 |
| Sanitization failures | 0 | 37 |
| Target-arm validity | 32 / 32 | 12 / 32 |
| Joint-arm validity | 32 / 32 | 15 / 32 |

The AE result shows valid categorical draws at these selected endpoints, with
little distinct chemistry beyond argmax. The VAE has lower validity here and
also very few distinct valid outputs. Neither result establishes an AE/VAE
advantage on independent seeds, useful binding changes or synthesis feasibility.
The starts are selected, correlated endpoints around the same 13-atom context.

Only **three AE draws and eight VAE draws are charge-neutral**. Other compatibility
checks may further limit that subset. Most of the planned GuacaMol projection
attempts would therefore be unsupported under the explicit charge-preserving
policy. Conditioning analysis on that small neutral subset would also change
the comparison population; keep the original 64-attempt denominator and state
eligibility separately. High raw validity does not resolve missing charge support.

The complete result bundle is now downloaded and all four run directories
extracted, retaining the original failed preflight, corrected tests and both
bridges. Local archive:
`outputs/hpc-results/stage2/stochastic-bridge-source025-results-20260928.tar.gz`;
HPC export:
`/work3/s234818/latentmol/exports/stochastic-bridge-source025-results-20260928.tar.gz`.
Matching local/HPC SHA256:
`a27e3fde84ef9b3c9fa45e86fd8abe37d90b77c3db6c1e8cb8aeb2c613fc8984`.
The local hash and both extracted bridge summaries were checked. These
outcomes do not establish phase-2 projection readiness.

## Frozen inputs and phase-1 budget

Use immutable source017 runs:

| Model | Saved pilot | Selected AE/VAE checkpoint SHA256 |
|---|---|---|
| AE | `20260928T073635Z_ae-boltz-latent-pilot_6889d56d` | `1b1d2a32aedfadb376d5a5e2d8802d574ed5c1d72851ed79517acaef418e9b95` |
| VAE | `20260928T073724Z_vae-boltz-latent-pilot_a8e9e0e3` | `80b22b9c5a5e9010457b49246f830f0425c5578458eab28ddf2a10f5f632cee9` |

Source017 archive SHA256 is
`06ef6636810f195b10ff0eea98c4458a297bf3c19685ded3532f96fa578b9686`.
The checkpoints were selected by validation validity at AE epoch 108 and VAE
epoch 94. Their training budgets differ; this study matches subsequent inference
budgets, not original training compute. The implemented bridge verifies the
checkpoint, latent-reference and prepared-context files against parent-recorded
hashes. It records hashes of the parent manifest, final attempts, baseline and
each selected trajectory, checks completed-parent protocol/trajectory metadata,
and requires saved hard endpoints to agree with parent records.

The coordinator's phase-1 subset is fixed **before seeing samples**:

- Arms: target and joint, in both models.
- Repeat: 0 only.
- Start indices: 0, 1, 4, 5, covering two radius-2 and two radius-4 starts.
- Eight categorical draws per endpoint: **8 endpoints × 8 = 64 draws per model**,
  **128 total**.
- Exactly the same **13 occupied slots** as the source017 prepared context.

Preserve each endpoint's saved argmax decode, own perturbed-start SMILES and
encoded-baseline SMILES. Parent files retain their original records; no new
baseline inference occurs. Write `selected_parents.json` with the selected
parent rows and trajectory hashes, then record that file's hash before sampling.
Keep repeated identities and arm memberships. Do not choose better-looking
starts or refill failed draws.

## Phase 1: categorical decoding with an explicit sampling law

`src/latentmol/boltz/stochastic_bridge.py` reads each endpoint's saved
`soft_fixed_occupancy_graph` directly. Its element probabilities already condition
the 13 occupied slots on a real atom; other slots are absent. At this hard fixed
occupancy, present-pair bond probabilities have no fractional occupancy factor
to remove. Sample the saved atom and charge categories and each unordered bond
once, mirror bonds, retain NONE diagonal/padding bonds and neutral absent charges.
Do not apply another occupancy factor. Recorded temperature is 1: this bridge
does not reconstruct raw logits or introduce temperature rescaling.

The CPU generator is reseeded for each draw by
`31000 + 1000*arm_index + 100*start_index + draw_index`, with target arm index 0
and joint index 1. AE and VAE use the same index/seed scheme and their own saved
probabilities. `sampled_graphs.pt` stores every drawn atom/charge/bond array;
`draws.json` and metric rows retain seeds, parent identifiers, argmax/start/baseline
comparisons, valid SMILES or failure detail, heavy-atom and charged-atom counts.
The bridge writes overall and per-arm summaries. No invalid draw or duplicate
is removed or replaced.

**Not recorded by this bridge:** original decoder logits, original NO_ATOM mass,
or sample log probabilities. The first two are absent from the saved
fixed-occupancy graph and require a separately identified re-decoding audit;
they cannot be recovered merely by relabeling its conditioned probabilities.
The original saved probability tensors remain traceable through trajectory hashes.

The implemented comparisons are valid identity equality to the saved argmax,
valid changes from own start and encoded baseline, unique valid identities,
charged-draw counts and detailed failures. Invalid identity comparisons remain
unavailable rather than being counted as changed molecules. A later audit may
separate valid-to-valid changes from rescues. Native-standardized identities
are not computed by phase 1; that mapping is required before native selection
because charged changes can disappear in preprocessing.

This phase has **zero new Boltz calls and zero new latent updates**. Sampling is
not itself a gradient estimator. No pathwise derivative through a categorical
draw, RDKit or a native sampler is claimed. A later straight-through or
score-function optimization study needs its own objective, estimator and tests.

## Phase 2 feasibility: what DeFoG actually provides

At pinned DeFoG commit `365bda9affadd5c2307014a0532ddaa244399441`, native
`sample_batch` starts from prior noise at time 0 and runs toward time 1; it does
not accept a candidate graph or partial starting time. The native conditional
step `sample_p_zs_given_zt` can supply transitions for a **new projection
wrapper**: corrupt a candidate with the checkpoint's forward kernel at a
declared time and continue toward the data endpoint. Native generation uses
categorical transitions under `no_grad`; this is not a differentiable manifold
projection or a promise to find a nearest valid molecule.
[Pinned sampler](https://github.com/manuelmlmadeira/DeFoG/blob/365bda9affadd5c2307014a0532ddaa244399441/src/graph_discrete_flow_model.py).

GuacaMol's five bond categories include aromatic bonds, making it the first
schema candidate; ZINC's four categories require explicit kekulization.
Neither has our formal-charge head or NO_ATOM semantics. A fixed node mask can
represent 13 actual nodes, but an absorbing noise category is not an absent
atom. **This limitation concerns the public upstream checkpoints, not DBMol's
adapted DeFoG.** DBMol Appendix C.3 describes an adaptation with nine elements,
formal charges and aromatic bonds; it reports projection from `t0=0.8` for
200 steps. The cached official GuacaMol weights are not that adapted model,
and a successful neutral interface check would not establish support for charged
source017 endpoints. Our proposed public-checkpoint wrapper and budget are not
a replication of that trained DBMol projector.
[DBMol, Appendices C.2–C.3](https://arxiv.org/pdf/2607.19237).

The official GuacaMol checkpoint is now downloaded at
`/work3/s234818/latentmol/cache/defog-official-guacamol/guacamol.ckpt`,
**255,167,415 bytes**, SHA256
`bc97d52ccf33a20d991f98349a1d37662273ab1c3e3f9777b938f9f144bc61f5`.
Acquisition and the later audited source028 load succeeded. The subsequent
interface smoke **failed its scientific gradient gate**: native soft auxiliary
features rejected the input, and the frozen-feature edge derivative did not
meet tolerance. No generation or candidate projection is established. The
setup failures and bounded diagnostic outcomes are detailed below.
[GuacaMol codec](https://github.com/manuelmlmadeira/DeFoG/blob/365bda9affadd5c2307014a0532ddaa244399441/src/datasets/guacamol_dataset.py),
[ZINC codec](https://github.com/manuelmlmadeira/DeFoG/blob/365bda9affadd5c2307014a0532ddaa244399441/src/datasets/zinc_dataset.py).

### Source026 environment and loader attempts

Immutable source026 SHA256 is
`81c361b02dc5759c6581ce721fab7e5840b130bd7ce1d0de47357a001b2a7f06`.
Its separate intended smoke uses one compatible neutral 13-atom training
control, not charged source017 candidates. It would compare native soft
auxiliary features with frozen hard features and check node/edge loss
derivatives at two finite-difference scales. That is an interface diagnostic,
not projection, a useful chemical score or a derivative through sampling.

| Job | Resources / purpose | Verified outcome |
|---|---|---|
| 29507280 | `hpc`, 2 CPU slots, 8 GB/slot, 2 h; isolated environment and strict CPU load | Failed before inference: missing `imageio` import from native visualization code |
| 29507284 | `hpc`, 2 CPU slots, 4 GB/slot, 30 min; official import dependencies | Completed successfully |
| 29507287 | `hpc`, 2 CPU slots, 8 GB/slot, 1 h; load retry after 29507284 | Failed before inference: strict loading rejected four `sampling_metrics.*` buffers |
| 29507288 | `gpua10`, 1 exclusive GPU, 4 CPU slots, 8 GB/slot, 2 h; dependent neutral smoke | Cancelled while pending after failed prerequisite; no numerical work executed |

The first failed run is
`20260928T123509Z_defog-native-env-setup_39fd5715`. Recovery installed
`imageio==2.31.1`, `hydra-core==1.3.2`, `overrides==7.3.1`, `PyGSP==0.5.1`
and `seaborn==0.13.2` in isolated
`/work3/s234818/latentmol/env/defog311`. The second load reached native model
construction; the rejected entries were non-network metric buffers because
the inference loader used `sampling_metrics=None`. A corrected loader is
planned to permit only those explicitly audited entries and verify equality
of every restored model tensor. It requires a new immutable release; no
successful recovery or smoke result is inferred here.

### Source030 follow-up diagnostic: completed, gate remains failed

Immutable source030 SHA256
`e74582ea1c3ca8e182e6a58f611c7de732e52d583719734d9fff577a7b1a32e0`
ran as **29507409**, now **DONE**, on `gpua10`: one exclusive GPU, four CPU slots,
8 GB per slot and a two-hour limit. The same neutral 13-atom control is retained.
The diagnostic captures the native-soft auxiliary-feature assertion traceback
and evaluates additional edge finite-difference scales **0.02/0.04**, alongside
the unchanged **0.005/0.01** gate. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` guards
against the unrelated plugin failure. **Four codec tests passed**. Run
`20260928T125551Z_stage3-native-defog-interface-smoke_c72694fc` completed with
**`gradient_gate_passed=false`**.

The exact native-soft failure is `extra_features.py:429`,
`assert (k4x >= -0.1).all()`, in k4 cycle features on fractional adjacency.
The unchanged native auxiliary-feature path does not support this tested
expected-soft graph. With hard auxiliary features frozen, the edge-direction
finite-difference errors were **4.55%, 4.92%, 1.70% and 11.86%** at epsilon
**0.005, 0.01, 0.02 and 0.04**, respectively. Node-direction errors at
0.005/0.01 were **0.016%/0.106%**. One additional edge scale below 2% does not
pass the declared multi-scale gate or establish tight local agreement. The
edge derivative remains unverified, and no DeFoG projection or useful-gradient
claim follows. Run/summary JSON for the original and diagnostic runs are now
downloaded under `outputs/defog-preflight/source028/` and
`outputs/defog-preflight/source030/`; source030 also includes scheduler stdout
and stderr. Local presence is confirmed, but no matching HPC/local archive
digest has been supplied for this transfer.

The native sampler fixes node count and has no NO_ATOM output vocabulary. Its
`p_xt_g_x1` helper accepts hard integer labels, not relaxed probabilities;
candidate-conditioned continuation therefore requires an explicit wrapper.
No occupancy/size optimization or differentiable categorical sampler follows
from a successful model forward or from a frozen-auxiliary-feature surrogate.

### Source028 audited load passed; completed derivative smoke failed its gate

Corrected immutable source028 was uploaded and its HPC SHA256 verified:
`56a11cbc14aa91d392cc8c545234e88449ec258ecae25228f176fb1664e06b0f`.
CPU job **29507352 is DONE**, run
`20260928T124510Z_defog-native-checkpoint-load-audit_6055f70e`. It used `hpc`,
two CPU slots, 8 GB per slot and a one-hour limit. The audited load confirmed
**696 native state tensors bitwise equal**, permitting only the four named
non-network sampling-metric buffers to be omitted. Output dimensions are
**X=12, E=5, y=0**; augmented input dimensions are **X=34, E=25, y=7**.
This resolves the loader failure without accepting unmatched model weights;
it is not yet a successful native forward or gradient result.

GPU smoke **29507370 failed before any tests or inference** because pytest
automatically imported a broken `platformdirs.pytest_plugin`. Failed test wrapper:
`20260928T124625Z_tests-stage3-native-defog-contract_d1be6642`.
This environment/plugin failure does not invalidate the successful CPU loader
audit and does not establish a DeFoG numerical failure.

Recovery GPU job **29507373 is DONE** on unchanged source028 with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. Resources remain `gpua10`, one exclusive GPU,
four CPU slots, 8 GB per slot and two hours. **All four codec tests passed**.
Interface run `20260928T124749Z_stage3-native-defog-interface-smoke_4d0f1680`
completed but recorded **`gradient_gate_passed=false`**. The neutral 13-atom
control was training index 9, `BrC1CCCCC1Cc1nccs1`, at time 0.8.

Native `compute_extra_data` rejected the expected-soft corrupted graph with an
`AssertionError`. Holding auxiliary features at their hard-input values allowed
a restricted forward/derivative diagnostic, with the following results:

| Frozen-extra input direction | Analytic derivative | Finite differences at two tested scales | Gate |
|---|---:|---:|---|
| Node category C → N | 0.0183537 | 0.0183508 / 0.0183731 | Passed |
| Edge category single → NONE | 0.00314746 | 0.00329763 / 0.00331029 | Failed: about 4.55% / 4.92% error, above 2% |

This is partial evidence about one node direction in an altered feature policy,
not verification of all graph derivatives, the complete native soft path or a
useful prior. The edge check failed at both scales. The checkpoint's audited
restoration remains valid, but **no full edge/native gradient verification,
source017 projection, categorical-sampler differentiation or Boltz combination
follows**. Results are coordinator-confirmed; no transferred-artifact receipt
for this smoke is recorded yet.

Native reconstruction can infer partial charges during molecule construction,
and `ChargeFeature` is a derived auxiliary feature. Neither supplies a learned
formal-charge output channel for lossless projection of our charged graphs.
A public checkpoint for DBMol's adapted charge-aware model was **not located
in the checked official sources**; this is not a claim that none exists.
An alternative-model comparison is recorded in
[20_chemical_prior_alternatives.md](20_chemical_prior_alternatives.md).

Required preflight before phase 2:

1. Load the acquired official weights; freeze checkpoint, upstream code,
   environment, native configuration and all feature/output-construction policies.
2. Prove graph-to-native and native-to-project label parity on compatible hard
   examples, including aromatic mapping, masks and exact fixed node count.
3. Reproduce native prior-to-data generation under matched settings/RNG, then
   test the candidate-start wrapper using the same transition implementation.
4. Establish that malformed but schema-compatible graph inputs are handled
   explicitly. Chemical invalidity must not be mistaken for vocabulary
   incompatibility; it is a potential repair target.
5. Profile cost and save representative trajectories before launching the full
   paired projection cohort.

Initially classify nonzero formal charges or other unrepresentable features as
**unsupported**, retaining them in the original 64-attempt denominator. Do not
silently set charges to zero. If a native-standardization or neutralization
branch is later adopted, it must be separately labeled with before/after
identities and its own budget; it changes the molecule being projected.
Many selected parent argmax endpoints carry formal charge, so compatibility
may be limited even before projection. Eligibility of the actual sampled draws
must be measured; a cached checkpoint does not resolve this representation gap.

## Phase 2: paired projection budget and retention

Apply the same frozen settings to **the saved 64 draws per model**, without
resampling to obtain eligible molecules. Every draw gets a projection-attempt
record: executed, unsupported, numerical failure, timeout or another explicit
reason. Pair the result with that exact raw sample and its parent latent record.

A revised initial proposal is **one projection per eligible draw using the final
100 transitions of the official 1,000-step grid**. GuacaMol's pinned experiment
uses `eta=300`, `omega=0.1` and `time_distortion=polydec`; an arbitrary 20-step
coarse grid need not preserve that transition behavior. Use the pinned schedule
to obtain the actual corruption time at the selected grid boundary rather than
silently treating grid fraction 0.9 as an undistorted time. A final-250-transition
suffix is a separately budgeted sensitivity, not an automatic retry.

At 100 transitions, 64 eligible draws would require **6,400 candidate-level
network evaluations per model**. Batched implementation can use fewer forward
calls; log both batch calls and per-candidate evaluations, batch size and wall
time. These are proposed settings pending native parity and cost profiling,
not a tested projector or established sufficient budget. Freeze final grid,
effective start time, stochastic settings and wall-time cap before inspecting
cohort outcomes. Do not increase retries for one model or one arm.

The three primary branches are endpoint argmax, categorical sample without
projection, and projection of the **same sample**. Match AE/VAE calls, candidate
quotas and termination rules; show actual wall time and failures separately.
Projection costs more than raw sampling, so equal raw draw counts alone are
not a computational-efficiency advantage. If its extra compute matters, compare
against additional raw samples under the same measured time budget as a
separately declared control, not an unrecorded rerun.

Report validity before/after, exact identity retention, changes, standardized
identity collapse, unique outputs, size, node/bond edit distances, fingerprint
similarity where both molecules are valid, structural alerts and runtime.
Also report how often distinct inputs collapse to one projected molecule.
No extra RDKit repair or largest-fragment salvage is allowed after projection
without a separately named intervention.

**Engineering screen:** mapping/native parity must pass; all attempted inputs
must be accounted for; projection must yield at least two distinct valid
standardized changed outputs across at least two selected start indices and
improve all-attempt validity over the paired raw draws to justify the proposed
native audit. Report every component and harms to initially valid samples.
This is a proposed minimum screen, not statistical efficacy or evidence of
chemical stability. Failure does not justify silently broadening the budget.

## Conditional fresh native Boltz audit

Before any score is obtained, freeze an equal per-model, per-arm and per-branch
selection policy and maximum number of candidates/calls. A small proposal is
**up to two unique eligible candidates per arm/branch**, chosen by a recorded
random seed or another explicitly predeclared score-independent rule. Final
quota and wall-time cap remain coordinator decisions before submission.
Underfilled branches stay underfilled. Record all selected, unselected,
unsupported and failed attempts; a selected-subset audit does not estimate
native success for every unscored sample.

Deduplicate by the **actual pinned Boltz-standardized input identity**, while
retaining original row and arm memberships. Predict fresh native poses for each
selected candidate, its valid pre-projection sample when applicable, its own
perturbed start, and encoded baseline. Include the argmax and raw-sampling
branches under equal selection rules. Reuse predictions only when standardized
input, target context, settings and seed match exactly.

Use the established source018 native policy as a proposed starting point:
pose seeds 17 and 23; 100 structure and affinity sampling steps; structure
recycling 0; pinned native affinity recycling 5; `bf16-mixed`; explicit MSA
subsampling to 128 rows. Verify the actual CLI behavior and record hashes of
both native checkpoints, target/MSA, code and effective standardized molecule.
These native calculations are distinct from the fixed-pose FP32 proxy; project
output must receive fresh features and geometry. Journal each call before
launch, retain a declared timeout and never silently replace a failed candidate.

For a selected candidate's **binary-score confirmation**, require both native
pose seeds to improve over both its own start and encoded baseline by a
predeclared positive tolerance, proposed `1e-4`. Report projected-versus-raw
differences separately when the raw molecule is valid; an invalid-to-valid rescue
cannot establish preservation of a previous native score. Record both Boltz
heads and disagreements. Native model predictions are not binding observations,
and the continuous head is not a universal affinity ranking for arbitrary
generated molecules.

**No automatic promotion:** confirmation must include distinct standardized
molecules, interpretable chemistry/retention and both controls, rather than
repeated aliases or a score gain that disappears after projection. A positive
result here remains one-context engineering evidence. Broader starts, new
contexts, variable size and gradient-based coupling require later protocols.

## Denominators and durable evidence

Keep a linked record chain: selected latent state → draw → eligibility decision
→ projection → standardized identity → native selection → each prediction.
For each model and arm report all 32 raw attempts, all 32 projection records,
executed projection calls, valid/unique/changed outcomes, native-selected and
unselected counts, complete paired seed comparisons, failures and actual cost.
Deduplication never reduces the raw-attempt denominator. Conversely, unselected
native candidates are **unmeasured**, not proven failures or successes.

Save immutable source/config and all checkpoint/state/cohort hashes, raw tensor
outputs, RNG metadata, conversion policies, effective identities, projection
trajectories, timing and errors. Add jobs/results to the ledger only when they
actually occur. Keep phase-1 sampling conclusions, phase-2 projection conclusions
and native-score confirmation distinct. **Variable atom count remains untested**
throughout this fixed-13-slot bridge.
