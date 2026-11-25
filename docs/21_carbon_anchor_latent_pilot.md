# Carbon-anchor latent pilot: protocol and results

Dated 28 September 2026. Immutable source031 extends
`src/latentmol/boltz/latent_pilot.py`. **CPU tests passed. The AE failed its
anchor derivative gate before optimization; the VAE completed 10 updates and
produced a valid changed hard molecule with a higher fixed-context binary score.**
Separate native hard-forward parity passed exactly, but the AE raw-affinity
chemical-gradient threshold failed before finite differences. A fresh-pose
endpoint audit's CPU preflight passed and static review found no blocker.
Native GPU rescore **29507658 is DONE: four predictions succeeded, but the
endpoint failed the two-seed binary-improvement confirmation**.
This protocol adds the explicit
`--start-policy carbon_anchor`; the existing source017 `perturbed_cohort` policy
remains unchanged. The prior C13 representation checks and separate native-pose
preparation are recorded in [docs19](19_carbon_chain_seed_preflight.md).

## Immutable release and current jobs

HPC source archive:
`/work3/s234818/latentmol/releases/source-031-carbon-anchor-pilot.tar.gz`.
Uploaded/extracted source SHA256:
`1dc612bedb5469c57f87325be06927679ec75eceacdf545816a568ba373056c9`.
The common prepared C13 snapshot SHA256 was verified at submission:
`fe91865528cfb4dbe3bfeb552d4e28b14a8ea5ea089cefe4ae4dd6f96f5d6f49`.

| Job | Purpose | Queue and declared resources | Status |
|---|---|---|---|
| 29507438 | CPU tests | `hpc`, 2 CPU slots, 4 GB/slot, 30 min | DONE: 17 passed in 17.53 s |
| 29507439 | AE carbon-anchor pilot, dependent on tests | `gpua10`, 1 exclusive GPU, 4 CPU slots, 8 GB/slot, 2 h | EXIT 1: anchor binary FD gate failed; no trajectory |
| 29507440 | VAE carbon-anchor pilot, dependent on tests | `gpua10`, 1 exclusive GPU, 4 CPU slots, 8 GB/slot, 2 h | DONE: anchor gate passed; 10 updates |
| 29507451 | Native hard-forward parity and raw-affinity FD, zero optimization steps | `gpua10`, 1 exclusive GPU, 4 CPU slots, 8 GB/slot, 1 h | EXIT 1: exact hard-forward parity; chemical-gradient threshold failed before FD |

CPU test run: `20260928T130245Z_tests-carbon-anchor-031_c7704cea`.
AE pilot run: `20260928T130321Z_ae-boltz-carbon-anchor-pilot_18114367`.
VAE pilot run: `20260928T130451Z_vae-boltz-carbon-anchor-pilot_a9376cee`.
Parity/raw-affinity run: `20260928T130729Z_boltz_fixed_pose_gradient_smoke_f349e8cd`.

Both GPU commands use `--start-policy carbon_anchor`, the shared C13 preparation
and their respective selected checkpoint/reference. Scheduler logs use
`/work3/s234818/latentmol/logs/{carbon-anchor-tests-031,ae-carbon-anchor-031,vae-carbon-anchor-031}.%J.{out,err}`.
The separate parity smoke uses `boltz.smoke --optimization-steps 0`, the same
C13 preparation and checkpoint; its logs use
`/work3/s234818/latentmol/logs/chain-native-parity-031.%J.{out,err}`.
Binary latent-objective checks and native hard-forward/raw-affinity checks
answer different questions and are reported separately.

The full source031 result bundle preserves the CPU tests, failed AE pilot,
completed VAE pilot, failed parity/gradient smoke and all logs. HPC export
`/work3/s234818/latentmol/exports/carbon-anchor-source031-results-20260928.tar.gz`
and local `outputs/hpc-results/stage2/carbon-anchor-source031-results-20260928.tar.gz`
have matching SHA256
`673f0b4822119a42d96cd68de6a788d5e6b1505eacf5690dbecb55770400a08d`;
the local archive hash was rechecked.

## Observed anchor and trajectory outcomes

The AE analytic binary directional derivative was **8.2346846e-12**. Numerical
central differences at all five declared scales were **0**, giving relative
error **1.0** and a failed gate. The exact starting chain was retained, but
**no target trajectory or random endpoint ran**. Its baseline binary probability
was **0.079036884**, raw affinity **1.887461066**. This does not show saturation
of the binary head. The cause of the near-flat latent response remains unresolved;
decoder/adapter sensitivity and numerical resolution need a separate audit.

The VAE analytic derivative was **0.0003076984**. Relative errors at epsilon
0.001/0.003/0.01 were **1.67%/0.084%/0.365%**, satisfying the required two-scale
gate. Its initial relaxed binary probability was **0.079057924**, with the hard
C13 baseline at **0.0790368840**. After **10 updates**, the hard endpoint was
valid **`CCCCCCCCCCC(C)C`**, a branched 13-carbon molecule; the random
equal-displacement endpoint remained the original linear C13 chain.

| VAE measurement | Encoded chain baseline | Target endpoint | Random endpoint |
|---|---:|---:|---:|
| Fixed-context hard binary probability | 0.0790368840 | 0.0926051587 | Same hard chain; no new identity |
| Relaxed binary probability | 0.079057924 | 0.0926627517 | 0.0790520459 |
| Relaxed raw affinity value | 1.8865900 | 1.7073256 | Not summarized here |

The run recorded **28 scorer forwards and 11 backwards**, with frozen parameter
and context digests unchanged. Its fixed-context hard-improvement screen passed
for **one target attempt**; the random control did not produce a new molecule.
This is one local engineering result. It does not establish a general VAE
advantage, fresh-pose score persistence or measured binding. The AE failure
remains part of the paired protocol and is not replaced by another start.

The separate raw-affinity smoke compared the native discrete reference and
adapter on the C13 context. Both heads matched **exactly**, with zero absolute
differences: raw value **1.887461066**, binary probability **0.079036884**.
At the AE anchor, finite gradient norms were **1.6227e-11** for the latent,
**6.2757e-13** for element logits, **3.7434e-23** for charge logits and
**8.8787e-14** for bond logits. The chemical-channel threshold **1e-12** failed
before any raw-affinity finite-difference test. Thus hard-forward parity passed,
but the smoke as a whole failed. This additional evidence supports a near-flat
AE decoder/graph path at this anchor; it neither proves a unique mechanism nor
indicates binary-head saturation. No full AE gradient pass is claimed.

## Question and prerequisites

Can each frozen, validation-selected AE/VAE decoder support a useful local
Boltz gradient from its **exact encoded 13-carbon chain**, without first moving
to a perturbed molecule? This is a one-seed engineering check, not an estimate
of design efficacy or a comparison of overall AE/VAE quality.

Use the selected AE epoch 108 and VAE epoch 94 checkpoints and their respective
training-only latent references. Their exact identities remain those in docs19.
Verify checkpoint, reference, dataset/split and graph-schema metadata, including
the 215,000-member training-reference count. Both runs must use the same newly
prepared native C13 context, native affinity checkpoint and recorded settings.
The source029 pose prerequisite has completed; the exact snapshot SHA256 is
`fe91865528cfb4dbe3bfeb552d4e28b14a8ea5ea089cefe4ae4dd6f96f5d6f49`.
A tyrosine pose is not a substitute. The chain-specific binary latent gates have
now run with the distinct AE/VAE outcomes above. Native hard parity passed,
while the AE raw-affinity channel threshold failed before its FD checks.

Require the native seed to decode as neutral `CCCCCCCCCCCCC`. Encode that seed
deterministically with each model, using the VAE mean rather than a posterior
sample. Require the decoder's hard **atom, charge and bond arrays** to equal
the native seed graph exactly. This is stronger than matching only the canonical
SMILES. An identity failure stops the run; latent inversion or a perturbed start
would require a separately declared intervention.

## Matched starting point, objective and budget

Optimize the standardized latent coordinate
`u = (z - training_center) / training_scale`, retaining each model's previously
fitted scale floor. The anchor is its encoded mean. AE/VAE normalization makes
the declared local step policy explicit; it does not make their geometries or
objective sensitivities equivalent.

Per model, declare **one anchor, one repeat and one target-only trajectory of
10 updates**, plus **one random matched-displacement endpoint**. There are no
radius-2/radius-4 starts, trust-only arm, joint arm or weight calibration in this
policy. Encoder, decoder and Boltz weights remain frozen. The pose, MSA/reference
features, ligand maps and 13 occupied slots remain fixed. Occupancy is conditioned
before the adapter; this experiment cannot change atom count.

The objective minimizes negative Boltz binary binder probability. Each update
follows the normalized negative objective gradient with step norm **0.25**, with
total standardized displacement clipped to **2.5** from the anchor. A nonfinite
gradient fails; a later gradient norm at or below `1e-12` produces no move and
must remain visible in the record. The continuous affinity head is also logged,
including any disagreement with the binary objective.

The random endpoint has the **same displacement norm as that model's target
endpoint** and uses the declared direction seed 29000. It receives relaxed and
hard fixed-context scores, but no gradient updates. This is a movement control,
not a control with equal oracle calls. AE/VAE have the same planned trajectory
and diagnostic budgets; actual forward/backward calls, skipped invalid hard
scores, failure-stage costs and wall time must be reported separately.

## Gradient gate at the unperturbed chain

Before any update, evaluate the exact binary objective's analytic gradient
**at the encoded anchor**. Require finite entries and norm greater than `1e-12`.
Check central finite differences along the normalized analytic-gradient direction
at epsilon values **0.0001, 0.0003, 0.001, 0.003 and 0.01**. At least **two distinct
scales must have relative error at most 2%**. Save all five results, including
failures, in `binary_gradient_gate.json`; do not retain only the most favorable
scale. A failed gate prevents the trajectory from running.

The AE's source027 chain decode had recorded graph log probability and
reconstruction loss of zero in finite precision. Exact reconstruction therefore
does not imply a sufficiently strong local gradient, nor does this diagnose
saturation of the downstream binary head. If the anchor fails this
gate, report that result without silently switching to a perturbed start or
altering temperature. A finite-difference pass is local derivative evidence,
not a chemical-prior or binding validation.

## Outcomes, denominators and evidence

Persist the baseline and all **11 target states** (initial plus 10 updates),
relaxed outputs, un-repaired hard graphs/SMILES, latent displacement, gradient
norms, both prediction heads and the matched random state. Score the hard graph
in the fixed context at baseline, trajectory start/end and random endpoint.
Report any invalid graph with its failure reason; no repair, replacement start
or extra random attempt is allowed in this budget.

The primary denominator is **one attempted target trajectory and one random
endpoint per model**, with any pre-trajectory gate failure stated separately.
A valid changed endpoint must differ from both its own start and encoded
baseline; here those controls are the same chain. The existing fixed-projection
screen additionally requires binary probability greater than both controls by
`1e-4`. Report each component even when the aggregate screen fails. An unchanged
hard molecule with an improved relaxed score is not a discrete design gain.

Retain immutable source/config, prepared-context, checkpoint/reference and
dataset/split hashes; `baseline.json`, `baseline_state.pt`, `final_attempts.json`,
trajectory tensors, `oracle_cost.json`, gate results, summary, run manifest and
all errors. Require before/after parameter and context digests to match. Count
diagnostic calls as well as optimization calls. Independent AE/VAE jobs may fail
at different gates; do not replace failed attempts to equalize completed runs.

## Source032 native endpoint audit: completed, confirmation failed

Immutable source032 is uploaded/extracted at
`/work3/s234818/latentmol/releases/source-032-carbon-native-rescore.tar.gz`,
SHA256 `157c9c4af41ac56bb8ed4bd33455025f0a47449d64fb332176b54cb964720a88`.
CPU preflight **29507614 is DONE** on `hpc`, two CPU slots, 4 GB per slot,
30 minutes, running `tests/test_boltz_carbon_native_rescore.py`,
`tests/test_boltz_native_rescore.py` and `tests/test_boltz_chemical_audit.py`.
Logs use `/work3/s234818/latentmol/logs/carbon-native-rescore-tests-032.%J.{out,err}`.
**Eight tests passed with two warnings in 7.96 seconds.** Static review found
no blocker. GPU native-rescore **29507658 is DONE on `gpua10`**, using
parent `20260928T130451Z_vae-boltz-carbon-anchor-pilot_a9376cee`, the source029
C13 template, and the declared paired seeds/100-step schedule below. Completed
run: `20260928T132239Z_vae-carbon-anchor-native-rescore_293a741c`.

**All four native predictions succeeded.** The native-standardized baseline
`CCCCCCCCCCCCC` and target `CCCCCCCCCCC(C)C` remained distinct 13-heavy-atom
molecules. The random hard control equals baseline, so its paired scores are
identical rather than independent predictions.

| Pose seed | Baseline binary | Endpoint binary | Endpoint − baseline binary | Endpoint − baseline continuous |
|---|---:|---:|---:|---:|
| 17 | 0.1256052404642105 | 0.07438764721155167 | -0.051217593252658844 | +0.5220773220062256 |
| 23 | 0.13756819069385529 | 0.1585942804813385 | +0.021026089787483215 | -0.1192050576210022 |

**`positive_binary_on_both_seeds=false`**; the criterion versus both baseline
and random also fails because those controls share an identity. The direction
reverses across native seeds. Both heads favor baseline at seed 17 and endpoint
at seed 23, accounting for the lower-is-favorable continuous value. The observed
fixed-context gain therefore **did not pass fresh-pose confirmation**. Four
successful CLI calls are operational success, not evidence of robust design
improvement. This remains one target endpoint in one protein context, with two
pose seeds rather than biological replicates.

The complete evidence bundle contains the GPU run, CPU preflight, logs and
source032 archive. HPC export
`/work3/s234818/latentmol/exports/carbon-native-source032-results-20260928.tar.gz`
and local `outputs/hpc-results/stage2/carbon-native-source032-results-20260928.tar.gz`
have matching SHA256
`0edd56dcbb9fa49a414fdbba399d12322c89503fb1ab6b9610f462d5448154c8`.
The local archive hash was rechecked.

The selection and execution protocol was:

`src/latentmol/boltz/carbon_native_rescore.py` accepts only the
completed source031 VAE carbon-anchor run with
its passed binary gate and exactly one target and one matched-random endpoint.
It freezes the **sole valid changed target endpoint plus the random control and
linear C13 baseline before native predictions**. There is no native-score-based
candidate selection or replacement. The source031 AE has no endpoint to submit.

Audit each selected input using the pinned native Boltz standardization path
before scoring, retaining its original and effective identities and any failure.
The endpoint must remain distinct from both effective controls for the strongest
confirmation. A collapsed identity is a failed distinct-candidate comparison,
even if a raw input string differs. Structural descriptors/alerts in this audit
remain diagnostics, not synthesis or stability proof.

Use native pose seeds **17 and 23**, **100 structure and 100 affinity sampling
steps**, one sample each, structure recycling **0**, pinned native affinity
recycling **5**, `bf16-mixed` and explicit **128-row MSA subsampling**. Each unique
submitted molecule receives the same two-seed schedule with fresh native
features/poses. In the observed selection, random equals baseline, so there are
**two unique molecules and four planned native predictions**. Reuse that control
prediction without pretending the duplicated role is an independent replicate.

Journal calls before launch, preserve timeouts/failures and report attempted and
successful predictions. Retain paired endpoint-minus-baseline and
endpoint-minus-random differences for **both Boltz heads and both seeds**.
The proposed binary confirmation requires a distinct effective endpoint and
strictly positive binary differences versus both controls at both seeds;
this native criterion is separate from the `1e-4` fixed-context screen. Report
the differences themselves, not only the boolean, and retain disagreement
between binary and continuous heads. Failed calls remain in the denominator.

Record parent run/source, selected checkpoint/prepared context, frozen-selection
inputs, target YAML/MSA, both native checkpoints and installed Boltz provenance
hashes, alongside all prediction/preprocessing/pose artifacts. The denominator
is one selected target endpoint, one control role identical to baseline, and
four attempted predictions, all completed here. Fixed-context scores cannot
substitute for this fresh-pose check, whose observed confirmation failed.

No native endpoint rescore, DeFoG coupling, synthesis/selectivity claim or
variable-size optimization is part of the source031 trajectory.
Even a positive local result remains a single-context engineering observation.
