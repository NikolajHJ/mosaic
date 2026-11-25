# Random-start latent optimization with increasing DeFoG guidance

**Latest verified snapshot, 10 October 00:10 DTU-local:** all ten long cases
failed (four 6,000-second timeouts and six return-code-1 outcomes); the
controller completed, but the requested cohort did not. The real six-update
repair/gradient gate remains passed. Source080 completed all three saved-state
replays, reproducing the Boltz FD mismatches with unchanged identities;
numerical cause remains unresolved and no original gate was reclassified.
See the [diagnostic report](47_defog_long_run_diagnostics.md) and
[ledger](05_experiment_ledger.md) for verified capsules and limits.
No code, scientific settings or jobs changed in this status retrieval.

**Historical investigation update, 9 October 2026:** native DeFoG repair and fixed-target
gradients worked in the short gate. Three long cases reached 60 accepted
updates and timed out inside the full Boltz gate; repeated component/combined
FD suites on every refresh dominate runtime. The cap was undercalibrated.
Separate failed Boltz directions still require numerical attribution; no
DeFoG-gradient failure is established and no production policy changed.
Source080 is sealed/uploaded; CPU **29661601 DONE** passed 31 fixtures in
12.94 s. Native **29661602 PEND**, four-hour A100/MPS saved-state replay,
awaits host slots with CPU dependency fulfilled. Real numerical attribution
remains pending; no production policy or tolerance changed. See
[diagnostic report](47_defog_long_run_diagnostics.md).

**Historical status, 9 October 2026 10:19 DTU-local:** source079 native **29656359
RUN**. The real six-update gate passed: eight eligible native repair targets,
two selected refreshes, six nonzero guided accepted updates and seven target-only
finite-difference passes. **No long case completed**: prior 0–5 failed (three
6,000-second timeouts and three exit-1 results), prior 6 is running, 7–9 are
unattempted. Follow-up logs show 0/3 halted at the Boltz `dbmol_contact_loss`
component FD gate and 5 at the conditional Boltz `local_v1` FD gate. Numerical
causes remain undetermined; these are not established DeFoG-gradient failures. This proves
the short repair/gradient path, not full-cohort stability or efficacy. The
[verified capsule](../outputs/hpc-results/status-20261009T081813Z.tar.gz)
and [docs43](43_defog_gpu_process_recovery.md) preserve evidence and limitations.

**Historical native-contract fix in progress, 8 October 2026:** static, remotely
hash-matched source inspection identified the Boltz singleton ensemble frame
axis and DeFoG one-hot `NONE` diagonal as adapter mismatches. The user authorized
fixing and verifying the run before resubmission. Source079 is sealed/uploaded;
CPU **29656209 DONE**, all 195 fixtures and actual saved 13-/22-atom native
frame-helper parity passed. Native **29656359 submitted/PEND**, one 80 GB
A100/MPS allocation for 23 hours at 16:55 DTU-local. Its allocation probe and
six-update native repaired-target/gradient/accepted-update gate must pass before
ten long cases; no successful real-model repair or efficacy result is yet
established. The scientific ramp,
frozen models and candidate/target policies are preserved. See
[docs43](43_defog_gpu_process_recovery.md#static-native-contract-diagnosis-and-correction-scope).

**Historical recovery startup passed; native contracts failed, 8 October 10:30
DTU-local:** source076 native **29634091 EXIT** after a successful repeated
A100 CUDA coexistence probe. The scientific gate failed
`Unsupported native confidence frame shape or coordinates`; all four
prescribed DeFoG attempts separately failed `native_continuation` with
`Native transition changed count/shape/symmetry or emitted nonfinite data`.
**None of the ten long trajectories launched.** Causes were not established
at that snapshot; no further fix/rerun was then claimed. Source076's 161
engineering checks and allocated negative/positive
MPS probes remain valid startup evidence. The new failure metadata was
observed over authenticated SSH and now downloaded/hash-verified in the
[source077 evidence capsule](../outputs/hpc-results/source077-dense-chemistry/evidence001.tar.gz). See
[docs43](43_defog_gpu_process_recovery.md) and the ledger.
The original failed study below and its source/config records remain retained.

Historical failure check, 8 October 2026, **09:45 DTU-local**: the user authorized an
additional random-initialization
experiment with a larger DeFoG repair-guidance weight near the end. The
config/controller/HPC scripts are sealed/uploaded as source069 attempt001.
CPU **29621174 DONE**, with **141 engineering checks passed** and eleven
prescribed configs prepared. Native **29621180 failed** the real six-update
preflight. The authenticated read-only recheck confirms unchanged failed
wrapper/controller/native-gate records; the job has aged out of LSF history.
Its original wrapper finished **7 October at 17:47:52 UTC**, return code 1. The nested
gate reports a DeFoG worker `CUDA-capable device(s) is/are busy or unavailable`
error, and the controller explicitly blocks the long cohort. **None of the ten
long trajectories launched.** The sole recorded metric is a schedule event
at step 0/effective weight 0.001, with no target hash, zero optimization oracle
calls and no recorded improvement. No accepted or guided update occurred.
One GPU in `exclusive_process` mode combined
with a CUDA Boltz parent and separate CUDA DeFoG subprocess is the strongest
static explanation; this cause remains unconfirmed. This is a
GPU availability/launch failure, not a negative chemistry or gradient result.
Verified failure logs and exact native receipt are downloaded in
[status evidence002](../outputs/hpc-results/source073-budget-recovery/extracted-status002/defog-wrapper/run.json),
with the archive/hash and current decoder progress recorded in the
[ledger](05_experiment_ledger.md). No new DeFoG rerun or fix was made; this
read-only status turn changed no code, jobs, VPN or key settings. No new
evidence bundle or numerical test was created.
The cohort is ten prescribed random seeds, at most 100 updates per seed,
executed serially, with exact historical saved starts and a late-half ramp
specified below. Implemented source/config/start identities, resource limits
and receipts are recorded below and in the ledger.
Existing source064/source065 and their outcomes remain separate historical
controls.

Source069 archive SHA-256 is
`405df53db752530dc67ff7322c5a46a363dd027b5e8f12840e9f4daeffbed8ff`,
scientific manifest
`d3740595b959184573d7249e3afac0b9ba02e8884b33fe9f01b67e68dd7837d3`.
Remote release is
`/work3/s234818/latentmol/releases/source-069-ramped-defog-random-attempt001`;
prepared/receipt roots are
`/work3/s234818/latentmol/submissions/source069-ramped-random-attempt001/{prepared,receipts}`;
campaign root is
`/work3/s234818/latentmol/runs/source-069-ramped-defog-random-attempt001-random-study`.
Prepared [protocol.json](../outputs/hpc-results/source069-ramped-defog-random/extracted/submissions/source069-ramped-random-attempt001/prepared/protocol.json)
SHA-256 is `43b578dd30378a9d659f78f28f1a788fca7bce3223481f1783e598da1ad0e1ac`;
it pins all eleven configs and every saved parent/start/native-template.
Guidance YAML SHA-256 is
`a0c9233261f8ad08dfadc87a3bf883d8df74bbbfa14ff7e3f73f79f5ec6e3f27`,
source063 starts protocol
`b4488b0422432f7de9613baea73cc245e610953e07d89aeb261e74bfc313a3d0`.
The [25-member downloaded evidence](../outputs/hpc-results/source069-ramped-defog-random/engineering-evidence.tar.gz)
SHA-256 is `7913cd85a3c5ffefe189ad9657d07b5711b0a6915144629cc9e15a890bd74c6d`.
The exact [native intent](../outputs/hpc-results/source069-ramped-defog-random/extracted/submissions/source069-ramped-random-attempt001/receipts/native-intent.json)
and sibling CPU intent/run metadata retain executed commands and environments.
The coordinator verified the completed CPU run and command log:
121 existing fixtures passed in 18.50 seconds, then 20 schedule/controller
fixtures passed in 16.61 seconds. These tests do not establish native repair.
The [ledger](05_experiment_ledger.md) retains every historical preparation,
failure and submission snapshot. Later docs-only status notes do not change
the sealed scientific source.

The implemented constant-target method is described in
[docs37](37_defog_repair_guidance.md). Source068 passed 121 HPC engineering
checks and prepared two six-update C13 configs; it did not run a native
DeFoG repair/guidance experiment. This new campaign must pass an actual short
native gate before its long trajectories are released.

## Question and fixed scientific boundaries

Does increasing the strength of a fixed repaired-target decoder loss improve
chemical proximity and raw output validity during random-start optimization,
while preserving useful Boltz signal? This is a hypothesis to evaluate,
not a consequence of engineering tests or a validity guarantee.

```text
prescribed random z -> frozen decoder -> current shared relaxed graph
  -> frozen Boltz objective
  -> decoder CE against current constant repaired reference
     with a predeclared accepted-update-dependent coefficient

between updates, outside autograd:
  prescribed hard draw -> frozen DeFoG candidate continuation
    -> audited compatible/valid target or explicit no-target outcome
```

The current shared graph is the actual Boltz input. The repaired graph is
an explicitly different guidance reference, held constant during a derivative
gate and line search. Decoder and external model parameters stay frozen;
only the molecular latent vector changes. Draws, DeFoG transitions, chemistry
inspection and target selection remain outside autograd. The resulting
decoder CE gradient is a surrogate with a fixed target, not a gradient of
DeFoG likelihood or of an expected repair operation.

## Prescribed random cohort and occupancy

Reuse the **exact saved prior-0 through prior-9 latent files** authenticated
by the source063 stability protocol. They come from the earlier unfiltered
Gaussian prior, not fresh draws or a subset chosen for chemical/Boltz outcomes.
The new protocol must preserve each original parent/file digest, RNG/prior
policy and start identity. Retain unsupported, invalid, failed and unstarted
cases in the prescribed denominator; do not replace them.

Boltz uses the previously requested **full DBMol composite objective** from
the source065 pipeline, with frozen affinity/confidence branches and the
same fixed initial-pose **44-residue pocket**. Freeze target/context, native
pose and pocket-anchor identities in each generated config. The branch
definitions and hard-input/component-gradient gates are in
[docs36](36_dbmol_composite_objective.md); selecting that config does not
certify its still-required native validation. The declared base is
`/work3/s234818/latentmol/submissions/source065-dbmol-attempt001/prepared/prior-0-dbmol-fixed_initial_mask.yaml`,
SHA-256 `c46802325aa0e8892cae9e56ce5f6af66fd06ec2b19abe8afe863fa809bed7b2`,
obtained from the actual file digest. The controller replaces its molecular
start only with each prescribed authenticated historical `z`; the native
carbon-chain template supplies protein/MSA preparation for fresh rescoring,
not a substitute molecular initialization.

Each trajectory fixes its initial occupied-slot mask. Different seeds can
have different initial counts, but occupancy does not change within these
trajectories: the current official GuacaMol repair path supports fixed-count,
neutral input with exact canonical/native label mapping. Charged input and
unrepresentable native output labels are explicitly rejected; no quiet
neutralization, slot permutation, atom deletion or candidate replacement is
allowed. Sampling/chemistry conversion cannot be hidden inside Boltz adapters.
Consequently, this first guidance campaign does not test adaptive atom count.

## Increasing coefficient and fixed-state evaluation

The declared coefficient starts at **0.001**, stays there through the first
half, then increases linearly to **0.1 on the last requested update**.
For `K` requested updates and `k` accepted updates before the next update,
the intended policy is:

```text
progress = k / (K - 1)          # for this K = 100 campaign
ramp = clamp((progress - 0.5) / 0.5, 0, 1)
coefficient = 0.001 + (0.1 - 0.001) * ramp
```

This uses accepted-update progress, not attempted trials, elapsed time,
chemical outcomes or reference success. For 100 updates, proposal indices
0–49 retain 0.001; index 99 uses 0.1. Each endpoint retains the coefficient
actually used at its final state; an early stall must not advance the
schedule to fabricate a high-weight endpoint. Record the actual coefficient
at every saved state, formula/version and requested budget. Small-budget
gate cases use their own declared progress denominator; their six-update
ramp is not evidence of 100-update behavior.

The authored schedule interface is
[`LinearRepairWeightSchedule`](../src/latentmol/repair_guidance/config.py)
inside `RepairGuidanceConfig.weight_schedule`. This is a **partial config
fragment**, not a complete runnable model/input config:

```yaml
repair_guidance:
  weight: 0.001
  weight_schedule:
    version: linear_v1
    start_weight: 0.001
    end_weight: 0.1
    start_fraction: 0.5
    end_fraction: 1.0
  normalization: sum
  refresh_interval: 10
  attempts_per_refresh: 2
```

Omitting `weight_schedule` preserves the older constant-weight policy and
identity fields. The schedule has a strict version/field contract and rejects
decreasing weights or unordered/out-of-range phase fractions. The runner
resolves one effective coefficient for a state and passes it explicitly to
all analytic, finite-difference and line-search evaluations. Its authored
`repair-weight-STEP.json` receipt records actual progress, coefficient,
target/point identity and schedule-induced objective jumps; the authored
implementation passed the 141-check HPC engineering stage; real native
validation remains pending at this document snapshot.

For an inner update, minimize

`Boltz loss(current graph) + current coefficient × decoder NLL(fixed repair target)`.

Use the same current coefficient, target, pose and occupied slots for the
baseline, all finite-difference perturbations and all line-search trials.
Neither trial index nor worker attempts may advance the schedule. The
coefficient is an externally prescribed scalar, not another differentiable
property of `z`. Preserve the unweighted reference NLL, weighted reference
term, Boltz terms, combined loss, branch gradient norms and alignment.

A coefficient change at unchanged `z` changes the objective even if the
reference is unchanged. A reference refresh can change it again. Audit these
two boundary changes separately from an accepted latent update, with old/new
coefficient, target identity, loss components and weighted jump. Do not claim
monotone descent across changing objectives. Backtracking establishes descent
only under its fixed state-specific coefficient/reference/context.

The reference NLL follows the established exact factorized decoder policy:
occupied real-element distributions exclude `NO_ATOM`; charges use the actual
target label; unordered bond decisions include `NONE` once. Default sum NLL
and mean-per-decision are different objectives, so the final protocol must
pin its reduction. A larger coefficient also changes gradient balance;
its numerical value alone is not an empirical measure of chemical validity.

## Refresh, rejection and bounded compute

Retain the existing explicit frozen-target contract. The long campaign
refreshes at accepted-update boundaries **0, 10, ..., 90**, with **two
prescribed repair draws per refresh**: at most 20 repair attempts per completed
100-update seed, 200 across ten completed seeds. Every draw remains recorded
even when an earlier draw succeeds. Unsupported input or early termination
can reduce actual native worker calls; the quota is not silently refilled.
The short gate uses refresh interval **five** and **four draws per refresh**,
so its six requested updates can exercise boundaries 0 and 5.
Candidate/native RNG streams, actual distorted start time/grid boundary,
sampler settings and source/checkpoint identities belong in the final config
and every native receipt. Use direct candidate continuation at distorted
time **0.8** on the pinned checkpoint's **1000-step full grid**, preserving
its actual sampler settings and recording the effective boundary/suffix
transition count.
All attempts are recorded in prescribed order; target selection cannot rank
Boltz scores or silently refill unsuccessful draws.

If no compatible valid target is obtained, clear the old target and record
the explicit Boltz-only fallback. Distinguish ordinary supported native/
chemical rejection from provenance, adapter and frozen-state contract errors,
which remain fatal. Raising the coefficient does not create a reference when
none is available, and an all-fallback trajectory is not evidence that native
guidance worked.

Report native worker attempts, denoiser forwards, attempted/completed suffix
transitions, wall time and censoring separately from Boltz forward/backward,
finite-difference, decoder and line-search work. One subprocess is not one
model forward. The worker exits before the next attached Boltz gradient,
but resident Boltz weights can overlap with its GPU memory use.

## Validation and serial execution gates

1. HPC engineering checks validate the strict schedule, serialization and
   fixed-coefficient behavior within one derivative gate/line search,
   together with the existing provider/codec/runtime fixtures.
2. A short **prior-0 six-update** native candidate/guided experiment verifies
   the actual checkpoint, isolated worker environment, exact graph mapping,
   continuation calls and a compatible valid repaired target. It uses four
   prescribed draws at each interval-five refresh. A positive guidance pass
   requires an accepted guided update, a nonzero weighted CE gradient and
   the prescribed target-only and combined finite-difference gates, as well
   as the required full-DBMol native/component gates.
3. Only after the required native gate passes may the serial ten-seed long
   campaign start. An accepted short smoke does not establish 100-update
   stability or chemical/binding improvement.

The authored controller/scripts and bounded resource plan are specified
below. Sealed/generated identities and submission receipts are recorded;
actual native gate/long-run outcomes remain pending. Preserve every
failed/preparation/submission attempt and immutable source/config identity
in the [ledger](05_experiment_ledger.md). A dependency or submitted job is
not a passed gate, and a controller that finishes early is not ten completed
100-update trajectories. There are no local numerical tests for this task;
engineering checks and model evaluations belong on DTU HPC.

## Authored HPC workflow and commands

[`random_study.py`](../src/latentmol/repair_guidance/random_study.py) prepares
exactly one prior-0 gate config and ten ordered long configs, retaining each
saved-start/parent/native-template identity. Preparation authenticates the
source063 starts protocol and full-DBMol base; it does not select or generate
new random vectors. The authored guidance-only example is
[`defog_ramped_random.example.yaml`](../configs/defog_ramped_random.example.yaml).
It is not a complete active-set config.

[`boltz_ramped_random_preflight.sh`](../scripts/hpc/boltz_ramped_random_preflight.sh)
runs the existing `boltz_repair_tests.sh` suite, then
`python -m pytest -q tests/test_repair_weight_schedule.py tests/test_ramped_random_study.py`,
before metadata preparation. Source069 CPU 29621174 passed 121 tests in the
existing suite and 20 in the new schedule/controller suite, then prepared
the eleven configs. These include mocked model/controller behavior. The
authored [`submit_ramped_random.py`](../scripts/hpc/submit_ramped_random.py)
verifies archive/source, base/starts/guidance identities and unique receipts;
without `--submit` it prints a plan. CPU template:

```text
python scripts/hpc/submit_ramped_random.py --stage cpu --release IMMUTABLE_RELEASE --archive-sha256 ARCHIVE_SHA --base AUTHENTICATED_BASE.yaml --base-sha256 BASE_SHA --starts SOURCE063_PROTOCOL.json --starts-sha256 STARTS_SHA --prepared-root UNIQUE_PREPARED_ROOT --receipts UNIQUE_RECEIPTS
```

After the matching CPU receipt is authenticated and its job is **DONE**, the
native submission requires the actual generated `protocol.json` SHA. It
adds `done(CPU_JOB_ID)` and runs
[`boltz_ramped_random_run.sh`](../scripts/hpc/boltz_ramped_random_run.sh):

```text
python scripts/hpc/submit_ramped_random.py --stage native --release IMMUTABLE_RELEASE --archive-sha256 ARCHIVE_SHA --base AUTHENTICATED_BASE.yaml --base-sha256 BASE_SHA --starts SOURCE063_PROTOCOL.json --starts-sha256 STARTS_SHA --prepared-root UNIQUE_PREPARED_ROOT --protocol-sha256 ACTUAL_PROTOCOL_SHA --receipts UNIQUE_RECEIPTS

python -m latentmol.repair_guidance.random_study run --protocol AUTHENTICATED_PROTOCOL.json --sha256 ACTUAL_PROTOCOL_SHA
```

These are generic command templates. The source069 CPU/native stages were
actually submitted, with exact arguments in their durable intent/reply/receipt
files above. CPU completion is verified; native is queued with
`done(29621174)` and has no result yet. All model work occurs in allocations;
the login helper performs only identity/submission operations.

| Stage | Requested resources | Declared cap/gate |
| --- | --- | --- |
| CPU fixtures/preparation | `hpc`, 2 cores, 4 GB/core, 8 GB host, one host | 30 minutes; must finish successfully before native submission |
| One serial native allocation | `gpua100`, `select[gpu80gb]`, one exclusive GPU, 4 cores, 16 GB/core, 64 GB host, one host | 23 hours; first run the real prior-0 gate, then the long cohort only on a positive guidance pass |
| Prior-0 native gate child | Within that same GPU allocation | 7,200 seconds, six requested updates |
| Each long optimization child | Same GPU, serial | 6,000 seconds, 100 requested updates |
| Each fresh-pose rescore controller | Same GPU, after optimization child exits | 900 seconds total for that case, not 900 seconds per pose |

The serial parent holds no model weights. Each optimization subprocess exits
before fresh-native evaluation and the next seed. The campaign's declared
controller budget is **79,200 seconds (22 hours)** within the 23-hour job.
Before starting a long case, reserve **6,960 seconds**: the full 6,000-second
optimization cap, 900-second rescore cap and 60-second boundary guard.
Insufficient remaining budget retains the remaining prescribed rows as
`not_attempted` with the allocation-boundary reason. There is **no automatic
continuation, seed retry or replacement**. A future authorized allocation
would require retained identities and an explicit continuation protocol;
the present controller does not claim one.

`campaign.json` retains the native gate and ten case rows. If the gate fails,
all ten long cases remain not attempted with `blocked_by_native_guidance_gate`,
and the controller raises an error. A failed long seed remains failed while
the remaining prescribed order can continue within the budget. A recorded
stalled trajectory retains its real endpoint and can be eligible for raw
endpoint rescoring; it is not a completed 100-update budget. Fresh-pose failure
is recorded separately from optimization completion. Therefore
`completed_requested_cohort` does not itself certify every native rescore.

## Required outputs and interpretation

Separate requested, attempted and accepted updates; completed, stalled,
failed and unstarted trajectories; and accepted, rejected, expired or absent
repair references. Record actual coefficient/refresh histories, step scales,
same-objective loss changes, objective-boundary jumps, frozen identities and
physical oracle costs.

Chemistry evaluation must retain raw hard outputs and the actual conversion
policy. Preserve per-seed start/terminal argmax and prescribed categorical
samples. Long cases retain **argmax plus eight prescribed hard draws at
each saved state**; a full 100-update trajectory has 101 states, up to 909
candidate memberships including argmax. The six-update native gate uses
argmax plus one hard draw per state. Report repaired-reference edits
separately from raw decoder validity; they are observed native-repair edits,
not an exact nearest-valid distance. The separate existing probability-aware
bounded witness audit can be applied to saved states later, but this serial
controller does not run that search or add its cost to the optimization loss.
No censored search would establish an exact nearest-valid distance. Do not
substitute a repaired target when claiming a raw optimized molecule is valid.

Assess variation among prescribed seeds using eligible raw chemical
identities and properties, with invalid/rejected denominators retained.
Latent closeness alone is not evidence of chemical similarity. The planned
fresh-native evaluation scores **eligible raw start/terminal argmax** for
each seed at native pose seeds **17 and 23**, with actual processed-input
identity audits. Invalid raw outputs retain explicit ineligibility and are
not replaced by repaired targets. There are at most 40 logical seed-outcome
memberships before invalid filtering/deduplication; actual model execution
and costs must be recorded. Until that evaluation completes, relaxed
frozen-pose objective gains remain proxy gains.

The first campaign tests whether the proposed ramp can execute and how its
trajectories behave. Establishing a benefit of the ramp itself requires a
matched constant-weight or zero-guidance comparison from the same initial
vectors, with declared differences in objective, occupancy and oracle budget.
Historical trajectories are contextual evidence, not automatically matched
controls for a new loss or new gate policy.
