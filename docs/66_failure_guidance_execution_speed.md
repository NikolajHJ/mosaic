# Measuring and accelerating failure-guidance execution

**Source129 awaits A100, 20 October 2026 18:16 DTU-local:** CPU/callback DONE,
diagnostic 29763843 PEND; no GPU result or training. Validity correction remains
unselected; startup/clock preparation is unsealed and static-only.
[Latest snapshot, accounting limits and next stages](68_validity_runtime_recovery.md#current-blocker-and-next-evidence-20-october-2026-1816-dtu-local).

**Source129 still PEND, 20 October 2026 18:05 DTU-local:** no GPU result or
training. Verified narrow cache cleanup completed; follow-up runtime/helper
work remains unsealed and changes no existing diagnostic source.
[Current wait, storage proof and scope](68_validity_runtime_recovery.md#latest-wait-and-separate-preparation-20-october-2026-1805-dtu-local).

**Source129 GPU diagnostic queued, 20 October 2026 17:57 DTU-local:** CPU
29763813/callback 29763814 DONE; probe 29763843 PEND on an 80GB A100 with
16 CPU slots/60 minutes requested. No GPU result or training is established.
[Actual receipt/configuration](68_validity_runtime_recovery.md#actual-bounded-gpu-diagnostic-submitted).

**Source129 CPU gate/profile passed, 20 October 2026 17:55 DTU-local:**
1,325 passed/70 CUDA-only skips, no failures/errors. Saved wrapper/allocation
completed, but LSF still reports CPU 29763813 RUN and callback 29763814 PEND;
no GPU diagnostic or training is queued.
[Actual CPU evidence and scheduler distinction](68_validity_runtime_recovery.md#cpu-gate-passed-scheduler-completion-not-yet-observed).

**Source129 staged, 20 October 2026:** CPU 29763813/after-CPU 29763814 are
submitted, not passed; no GPU probe or training is queued. The release is
diagnostic only, with no selected validity fix or optimizer-step policy.
[Actual source/plan/receipt and commands](68_validity_runtime_recovery.md#source129-actual-staging-and-cpu-submission).

**New recovery scope authorized, 20 October 2026; preparing:** address validity
and runtime together using a new immutable candidate from epoch 36, with all
gates unchanged. No optimizer-step policy, new job or pass is yet established.
[Targeted investigation and validation requirements](68_validity_runtime_recovery.md)
preserve source127/source128 history below.

**Source128 native failed, 20 October 2026 17:19 DTU-local:** runtime exceeded
the unchanged cap, and the eight-update panel also shows a retention drop
beyond the supplementary limit (gate not assembled). CUDA/gradient fixtures
passed; no 2,500-update proof or training started.
[Saved outcome and limits](67_guarded_failure_guidance_restart.md#source128-native-benchmark-failed-no-stability-proof-or-restart-20-october-2026-1719-dtu-local).

**Source128 full CPU passed, 20 October 2026 09:25 DTU-local:** CPU
29760552/after-CPU 29760553 DONE, repair profile passed; 1,252 unit passes,
66 CUDA-only skips, no failures/errors. Native/stability GPU 29760576 and
callback 29760577 are PEND; training is not submitted/restarted.
[Actual proof, resource wait and metadata-collector failure history](67_guarded_failure_guidance_restart.md#full-cpu-gate-passed-nativestability-gpu-queued-20-october-2026-0925-dtu-local).

**Source128 allocated unit proof, 20 October 2026 09:22 DTU-local:**
1,252 passed/66 CUDA-only skips, zero failures/errors. CPU 29760552 remains
RUN in its profile phase with no saved full `cpu_gate`; callback 29760553
is PEND. GPU validation and training are not queued.
[Proof and command scope](67_guarded_failure_guidance_restart.md#allocated-unit-tests-passed-full-cpu-gate-pending-20-october-2026-0922-dtu-local).

**Source128 actual submission, 20 October 2026:** a new immutable release
authenticated 580 members and submitted CPU 29760552/after-CPU 29760553,
not passed. GPU native/stability validation and training are not yet queued.
[Protocol and receipts](67_guarded_failure_guidance_restart.md#source128-sealed-and-cpu-submitted-20-october-2026)
record the new objective and disposable proof; source127 stays stopped.

## New corrective validation authorized; preparation only, 20 October 2026

The user authorized versioned partial-loss scaling, early fixed-panel health
checks and separate chemical/representation diagnostics. The
[new staged protocol](67_guarded_failure_guidance_restart.md) requires a longer
DTU HPC update proof from epoch 36 through the earlier deterioration window,
all existing scientific/runtime/quota checks, and discarding the test state
before any conditional fresh continuation. No new source/job/result is yet
established. This authorization permits only a new passing immutable plan;
source127 remains stopped and its evidence below remains unchanged.

## Saved-evidence investigation after the user stop, 20 October 2026

[Analysis 002](../outputs/hpc-results/source127-bond6-runtime-continuation-attempt001/stopped-analysis002.json)
analyzes **24 saved training rows** and authenticates five reviewed source
modules against the sealed training manifest. The
[full stopped evidence](../outputs/hpc-results/source127-bond6-runtime-continuation-attempt001/stopped-evidence001.json)
retains complete logs and the 19 October 22:37:22 UTC scheduler observation:
training 29758907 and callback 29758908 are both **EXIT**, with an empty
checkpoint inventory. Analysis 001 and every failed/stopped record remain
preserved. Epoch 36 is the last completed checkpoint; epoch 37 has no saved
weights, completed fixed-prior panel or held-out reconstruction result.

The starting epoch-36 fixed 1,024-prior panel had **641/1,024 raw-valid
argmax graphs (62.5977%)**, and its held-out reconstruction panel had
**256/256 exact and valid reconstructions**. These are not comparable to the
changing 512-prior training batches' teacher-already-valid counts. The teacher
also requires schema `encode_smiles` and canonical roundtrip acceptance;
`population_panel` uses `inspect_graph` plus `checked_molecule`. Thus the
logged **0/512 teacher-already-valid** at update 239700 does **not** establish
zero raw RDKit validity. Counts fell from 377 at update 239100 through
345, 117, 19, 5 and 2 to zero, then were 9/512 at update 239800. The final
row is 2,344/6,596 updates (35.5367%) into epoch 37, with 486/512 original
search-censored rows. These are sampled training draws, not a matched
population evaluation or evidence of recovery.

No decoded graphs or epoch-37 weights were saved. Per-row teacher receipts
survive only through their hash chain, so the saved evidence cannot separate
native chemical failures from schema/roundtrip rejection, identify the
predominant atom/bond defect, or establish a sulfur change. All logged main
losses and gradients are finite, but that does not establish scientific
health. Review found no sign, source, train-mode or microbatch bug in the
examined code/evidence; this does not prove the absence of all defects.

The review establishes an objective-design concern, not a causal diagnosis.
Complete-target NLL averages `N + K + K*(K-1)/2` labels: 303 for `N=50,
K=22`. Partial-target NLL averages only the 1–6 edited bond labels, while
both configured component weights are 1. At update 239700, the weighted
partial contribution was approximately **0.0293294**, complete plus extra
valid-witness contributions **0.0002381**, and local-risk contribution
**0.0117761**. These quantify objective-value allocation, **not whole-branch
gradient ratios or proof of the deterioration's cause**. Already-valid prior
rows contribute zero repair loss; the occupancy anchor references epoch 35,
and population JS does not protect bond structure. The new objective retained
the epoch-35 repair ramp, giving coefficient 1/3 at epoch 37 (and prior-branch
outer factor 0.75), rather than starting a new ramp. Fixed-panel health checks
were scheduled at the 6,596-update epoch endpoint, which was not reached.

Recommendations before any proposed restart are consistent partial-target
normalization, an earlier fixed-panel guard, and separate raw-valid versus
schema-roundtrip logs. **No recommendation is adopted here: no loss change,
restart, new model test or new submission occurred, and automatic
continuation remains stopped.**

The root executed the following command tails using bundled CPython at
`C:/Users/nikol/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`
from the repository root:

```text
outputs/hpc-submissions/stop-source127-attempt001.py
outputs/hpc-submissions/collect-source127-attempt001-status.py --tag stopped001
outputs/hpc-submissions/collect-source127-attempt001-status.py --tag stopped002
outputs/hpc-submissions/collect-source127-stop-evidence.py
outputs/hpc-submissions/analyze-source127-stopped.py
outputs/hpc-submissions/analyze-source127-stopped.py --tag 002
```

The stop command performs the authorized scheduler cancellation; collectors
read scheduler/saved metadata, and analysis uses standard-library saved-log,
JSON, source/hash checks and arithmetic. This is not local numerical model
validation or a fresh evaluation.

## User stop confirmed; partial epoch 37 preserved, 20 October 2026 00:36 DTU-local

[User-stop receipts](../outputs/hpc-results/source127-bond6-runtime-continuation-attempt001/user-stop-20261020-attempt001.json)
record successful `bkill` return code 0 for callback 29758908 first
(19 October 22:36:11 UTC), then training 29758907. The
[22:36:54 UTC stopped snapshot](../outputs/hpc-results/source127-bond6-runtime-continuation-attempt001/status-stopped002.json)
confirms both **EXIT**. Training terminated at 22:36:20 UTC/20 October
00:36:20 DTU-local with `TERM_OWNER` and exit 130. KeyboardInterrupt is
expected evidence of the user-requested termination, not a spontaneous bug,
model-instability finding or scientific gate failure.

The final saved row is epoch 37/update **239800** at 22:34:47 UTC:
**2,344/6,596 updates (35.5367%)** since epoch 36. Its changing 512-prior
batch has 9 teacher-already-valid rows and 486 original-search censored rows,
after the preceding sample had zero already-valid. These are different
training draws, not a fixed-panel recovery or completed epoch. No epoch-37
checkpoint/fixed-panel health result is observed. The completed epoch-36
checkpoint remains **47,664,990 bytes**, SHA256
`3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f`,
verified unchanged. Preserve all partial logs/data/optimizer/RNG lineage;
**no automatic continuation or resubmission is authorized during this stop**.

Investigation has raised a normalization-imbalance hypothesis: complete
valid-witness NLL averages over roughly 303 categorical labels for about
22 occupied atoms, whereas partial NLL averages 1–6 edited bonds at the same
declared weight. This could change relative gradient scale alongside fallback
terms, but it is **not a proved trigger/root cause**. Do not encode a fix,
attribute causality, or alter the sealed objective from this hypothesis.
The fixed epoch-36 evaluation and sampled epoch-37 teacher deterioration
remain distinct evidence; further full-log collection is read-only.

## User requested stop and investigation; actual stop receipts pending, 20 October 2026

The user explicitly authorized **Stop and investigate**. Root is stopping
exact training job 29758907 and callback 29758908; preserve the instruction
separately from execution and wait for actual receipts before marking either
canceled. The earlier automatic-continuation authorization is suspended by
this current stop instruction for this branch. No other HPC submission is
established, and all partial scientific evidence/lineage must remain intact.

The [last saved status](../outputs/hpc-results/source127-bond6-runtime-continuation-attempt001/status-news002.json)
is 19 October 22:32:39 UTC (**20 October 00:32 DTU-local**), still RUN before
stop confirmation. Latest saved progress is epoch 37/update 239700 at
22:10:19 UTC/00:10:19 DTU-local: **2,244/6,596 updates (~34.02%)** since
the completed epoch-36 parent. No completed epoch-37 checkpoint or fixed-panel
evaluation is observed; epoch 36 remains the completed baseline.

The epoch-36 [fixed saved evaluation](../outputs/hpc-results/source123-retirement-recovery-002/source-latest_epoch.json)
has raw argmax validity **641/1,024 (62.60%)** on the fixed prior panel and
held-out reconstruction **256/256 exact and valid (100%)**. In contrast, the
changing 512-prior training batches' original-teacher already-valid counts
declined from 377 at update 239100 through **345, 117, 19, 5, 2 to 0** at
update 239700; latest original search is censored for **509/512** rows.
Teacher already-valid requires native validity plus schema roundtrip and is
not the same metric/cohort as fixed-panel raw argmax validity. Do not present
this as a matched epoch-36 versus completed-epoch-37 panel comparison or a
measured causal mechanism. It is saved evidence warranting the authorized
investigation.

All logged losses/gradients remain finite, but that does not establish
scientific health or resolve the teacher-validity deterioration. No epoch-37
validity improvement, healthy endpoint, completed training or confirmed
cancellation is claimed. Preserve original failures, partial logs and model/
optimizer/native+fresh-prior RNG identities for investigation.

## Training RUN with partial epoch-37 progress, 19 October 2026 18:03:33 UTC

The [fresh saved scheduler/run snapshot](../outputs/hpc-results/source127-bond6-runtime-continuation-attempt001/status-news001.json)
confirms **training 29758907 RUN**, started **16:37:56 UTC/18:37:56 DTU-local**
on `n-62-18-10`; callback 29758908 is PEND. Its active-phase record names the
correct job, and allocation/numerical-run records are running. The allocation
is one 80 GB A100/16 CPU slots/24h, using about 4.5 GB host RAM and zero swap
at this read. A start is now established, superseding the queued-only stage.

The [saved progress summary](../outputs/hpc-results/source127-bond6-runtime-continuation-attempt001/progress-news001.json)
is metadata analysis of **nine existing stdout metric rows**, not a new
numerical test or full-epoch evaluation. The latest row is 17:57:50.462507 UTC,
epoch 37/update 238300: **844 completed updates since epoch 36 out of 6,596
for the epoch (~12.796%)**. This is an update fraction, not a wall-time or
100-epoch-campaign completion percentage. Epoch 36 remains the latest completed
authenticated checkpoint; no completed epoch-37/fixed-panel health result exists.

All sampled total/reconstruction/KL/repair/main-gradient values are finite,
encoder/decoder gradients are positive, all nine repair probes are positive,
and the seven present failure-only probes are positive. The other two probes
are `None` at updates 237700/237900 because there are zero original failed
rows, matching the existing probe condition; that is not an observed failed
gradient check. All nine rows retain 256 real/512 prior points, with zero
sampled clipped updates and zero sampled search errors. These observations
cover the logged samples, not every update or a population evaluation.

The latest batch has total loss 0.0639478316, reconstruction 0.0038430172 and
KL 53.6273518; these are individual batch values, not epoch means or chemistry
quality conclusions. No code/job changes or model reruns were made for this
status check, and no completion ETA is inferred solely from these samples.

## Source127 CPU/adoption passed; actual training queued, 19 October 2026 16:34:44 UTC

The [saved 16:34:44.253427 UTC snapshot](../outputs/hpc-results/source127-bond6-runtime-continuation-attempt001/status-cpu002.json)
confirms **CPU 29758886 DONE (164s)**, full contracts passed, and
**after-CPU 29758887 DONE (15s)**. JUnit collected 1,233 cases: **1,168
passed, 65 CUDA-only skipped, zero failures/errors**, suite time 156.407s.
This includes 107 new CPU contract passes. Derived evidence-only adoption
passed using authenticated completed source126 science plus the explicit
factor-1.25 policy; no new GPU benchmark or scientific model rerun occurred.
Source126's old runtime-failed records remain unchanged.

`advance-after-benchmark.json` records `training_submitted_not_passed`,
completed parent epoch 36, actual **training job 29758907** and **callback
29758908**. Both are PEND, with 16 CPU slots reserved on `n-62-18-10` and
normal new-job scheduling/no ineligible wait. Training requests one exclusive
80 GB A100, 16 CPU slots/64 GB, **24h allocation/23h internal**. The callback
uses `ended(29758907)` and authenticates completion/health before further
serial continuation. **This is queued training, not started training, and
no epoch 37 completion is established.**

Fresh quota before segment 001 passed at **10.82 GiB free** against
**10.0625 GiB required**. The plan retains the authenticated epoch-36
model/AdamW/native+fresh-prior RNG, reference35/ramp, 256 real/512 prior
(1:2), loss/search/weights, 100-total-epoch maximum and every non-runtime
scientific/health/retention/resume/finite/memory/quota guard. Derived gate
SHA256 is `58f47898fcf3530ee91d0236459c9e83af9c0ef37184ac0668ad5de83fe78961`;
failure-readiness SHA256 is
`6781e89b67bdddce783edfb124bb354a41333eeda966a30e42a4d9269ec5c088`.
Actual submission receipts in the snapshot bind both hashes and the sealed
source127 archive/manifest/plan below. All prior authentication, execution,
retention and runtime failures remain preserved.

## Source127 authenticated staging succeeded; allocated CPU submitted, 19 October 2026

The user restored fresh SSH access. The
[upload/staging retry](../outputs/hpc-submissions/source127-bond6-runtime-continuation-attempt001-upload-stage-retry001-output.json)
authenticated **569 members** of the unchanged sealed source127 release.
Actual **CPU 29758886/after-CPU 29758887 were submitted**, one hour/16 CPU
slots, as recorded in the [queue output](../outputs/hpc-submissions/source127-bond6-runtime-continuation-attempt001-queue-output.json).
Preserve the initial password-authentication failure below; no archive or
scientific bindings changed for this transport retry.

This CPU stage performs evidence-only adoption and allocated contracts, with
no new GPU benchmark. The [16:31:46 UTC snapshot](../outputs/hpc-results/source127-bond6-runtime-continuation-attempt001/status-submitted001.json)
shows **CPU 29758886 RUN/16 slots, callback 29758887 PEND**, and
`failure_readiness.status=passed`: additive derived adoption/gate creation
authenticated the old science and new runtime policy. **CPU contracts are
not finished and training is unsubmitted.** This derived metadata pass is
not a new GPU scientific pass. Quota is 89.16/100 GiB used, 10.84 GiB free.
Only passed contracts, authenticated saved evidence/all
unchanged non-runtime checks, the declared factor-1.25 runtime policy and
fresh quota may trigger the planned automatic 24h allocation/23h-internal
serial training from epoch 36 toward 100 total. Actual outcomes and training
receipts will be recorded when observed.

## Source127 sealed; first upload rejected before authentication, 19 October 2026

[Source127 preparation](../outputs/hpc-submissions/source127-bond6-runtime-continuation-attempt001-preparation.json)
and [plan](../outputs/hpc-submissions/source127-bond6-runtime-continuation-attempt001.json)
bind archive `7bc32aef45938bf7f37f7543335ac3c54090f3b6fcd8568c9a144b9486c802c0`,
manifest `6141be0adcb3d6297332806d006a13ad2f39f06ad15ba7de8671043d61148aeb`
and plan SHA256 `5d3c5503eb0adf01e66549d35b2b5679f598cb23b06f14070f023f8b64d33fb6`.
There are ten exact code/test/controller overlays, 506 manifest files and
569 expected archive members; these are local packaging counts, not remote
staging validation. Independent static review passed. The plan binds 44 saved
artifacts (43 metadata records plus suite); its
[source126 evidence snapshot](../outputs/hpc-results/source126-loss-validation-speed-attempt001/runtime-adoption-evidence001.json)
SHA256 is `c6e835866393e9b3364c863d82c828837cca6d33e87f78bd193b477455ba5212`.
The runtime factor/reserve and evidence-only adoption scope below are unchanged.

The [first upload/staging output](../outputs/hpc-submissions/source127-bond6-runtime-continuation-attempt001-upload-stage-output.json)
records SSH return code 255, `Permission denied (password)`, before remote
control mkdir, upload or submission. **No source127 upload, job, allocated
contract pass, derived adoption pass or training exists at this entry.**
Fresh SSH readiness was requested; VPN/SSH settings were not changed and no
repeated connection was made. Preserve this transport failure separately
from scientific checks. The sealed adoption is ready pending access and its
required allocated CPU contracts/fresh quota; it is not a new GPU run.

## Resolved runtime allowance; source127 metadata adoption preparation, 19 October 2026

The resolved policy is a **top-level immutable plan field**:

```json
{
  "runtime_policy_amendment": {
    "version": "chemistry-epoch-fit-amendment-v1",
    "factor": 1.25,
    "reserve_seconds": 1800
  }
}
```

Scientific protocol/configuration bytes and 1:2 sampling remain unchanged.
Source126 mean projection is 54,516.4273s; its slowest 9.29592099s update gives
the saved-data conservative extrapolation **61,914.5513s (~17.1985h)**.
The new requirement `1.25 * T + 1800` is **79,193.1891s (~21.9981h)**,
fitting the same **82,800s (23h) internal cap** with **3,606.8109s (~1h)
margin**. LSF remains at most 24h. The lower multiplier is the explicit
user-authorized runtime policy change; it does not change an objective,
retroactively pass the old 1.5 rule, or measure a full training epoch.

Source127 is being implemented as additive **metadata-only adoption plus
allocated CPU contracts**, not a new GPU benchmark. It authenticates the
original source126/native 104-event, 22-loss and 13-parity evidence, all
17 original non-runtime checks, source/model/optimizer/RNG provenance and
the eight saved updates, then recomputes 16 supplementary non-runtime checks
from the saved panel/update evidence. A derived adoption receipt must label
its original source/provenance and new policy explicitly while preserving the
failed old benchmark/selection/promotion records.

The new trainer preflight uses the same factor/reserve and conservative
estimate, with every existing 900s phase deadline, native/scientific/health/
retention/resume/finite/memory/identity/quota and serial-continuation check
retained. Only successful allocated CPU contracts and fresh actual quota can
permit the planned automatic 24h training submission from epoch 36 toward
100 total epochs; 23h internal/epoch-boundary rules remain. **Source127 is
not yet sealed and no source127 job, passed readiness or training submission
is established.** Exact source hashes and receipts follow actual execution.

## User-authorized runtime-policy amendment; adoption preparation, 19 October 2026

The user explicitly requested starting the current bond6 VAE chemistry
continuation despite the runtime difference. This authorizes a **scoped new
runtime allowance** for a new immutable adoption; it does not retroactively
pass source126's failed gate, selection or promotion. Preserve every original
record and use an additive adoption identity that authenticates the completed
native evidence and unchanged scientific bindings. The exact factor,
implementation/source identity and jobs are not established yet.

Source126's mean projection is 54,516.4273s (15.14345h), 516.4273s above the
historical protected-estimate cutoff. The separate slowest update was
9.29592099s; the unfinalized slower-update extrapolation is about **17.2h**,
not the mean or a finalized supplementary result. Both estimates must be
declared and considered when specifying the amended runtime allowance. Do
not describe this solely as a nine-minute actual-epoch overrun; no training
epoch was measured. Keep the **24h allocation/23h internal** limits and
honest partial/epoch-boundary behavior regardless of allowance.

Authenticate all unchanged non-runtime checks from their actual saved
evidence before adoption: native/source/plan/checkpoint identities, scientific
and retention checks, gradient/parity/finite/memory/health and exact resume,
plus fresh official quota and no competing training/controller. Retain the
epoch-36 model/AdamW/native+fresh-prior RNG, frozen epoch-35 reference/ramp,
256 real/512 prior (1:2), coefficients/search/normalizations and maximum
100 total epochs. The runtime amendment does not admit fallback's failed
validity-retention result or relax another scientific requirement.

The root is reviewing minimal metadata adoption/reuse of completed evidence;
no broad rebenchmark, new source job or training submission is established
by this entry. Actual policy/hash/readiness/submission receipts will be
recorded when confirmed. [AGENTS.md](../AGENTS.md) now records this scoped
user override while preserving the historical gate policy below.

## Source126 native comparison finished; original runtime gate blocks bond6, 19 October 2026 16:01:46 UTC

The [fresh saved result](../outputs/hpc-results/source126-loss-validation-speed-attempt001/status-news002.json)
at 16:01:46.205918 UTC records selection `stopped`, reason
`no_passing_new_objective`, no winner and promotion
`stopped_no_passing_new_objective`. Native 29755832's allocation completed
14:19:23–14:34:23 UTC with return code 0; saved LSF logs report successful
native/callback completion (906s/9s). Their scheduler records are now aged
out, so absence alone is not the completion proof. **No training was submitted.**

All **104 deterministic event tests and 22 loss-validation tests passed with
zero failures/errors/skips**, with JUnit suite times 12.965s and 11.361s
respectively. The 13 native parity checks and 22/22 observed failed-row extra
gradient checks passed, including native VJP/frozen-target finite differences.
This establishes execution/parity/observed gradient behavior; full readiness
still requires every retention/runtime check.

| Arm | Original native gates | Supplementary outcome | Mean epoch projection | Conservative epoch projection |
| --- | --- | --- | ---: | ---: |
| `legacy3` | All 18 passed | All passed; control only | 36,850.6092s (~10.2363h) | 39,473.1604s (~10.9648h) |
| `fallback` | All 18 passed | Only `raw_prior_validity_retention` failed | 45,461.6191s (~12.6282h) | 50,098.0123s (~13.9161h) |
| `bond6` | Only `complete_epoch_fits` failed; other 17 passed | Not finalized after original gate failed | 54,516.4273s (~15.14345h) | Absent, not zero |

Bond6 exceeds the unchanged **54,000s protected-estimate limit by
516.4273s** (about 8m36s) at the original mean-based gate. It did not reach
the supplementary conservative estimate. Do not carry forward source125's
15.15376h conservative value or substitute zero for this missing result.
The unchanged requirement remains `1.5 * T + 1800 < 82800`. These are
eight-disposable-update epoch projections, not measured training epochs,
resource crashes or permission to relax the safety margin.

The fixed 256-point raw-prior panel still starts at 149 valid and ends
160/131/172 for legacy/fallback/bond6. Fallback's 149→131 decline fails
the two-percentage-point retention limit despite passing runtime. Bond6's
149→172 result and 12 extra valid/10 invalid partial witnesses among 22
original failed panel rows remain promising short observations, but do not
make that arm promotable or identify a cause for fallback's validity decline.
No long-run efficacy or epoch-100 endpoint is established.

The saved source125→source126 comparison has **exactly equal non-timing
metrics for all eight updates in all three arms**, excluding only
`prepass_seconds`, `search_seconds` and `update_seconds`; updated population
and reconstruction records are also exact. Both allocations used
`n-62-18-11`. For bond6, mean update/search/prepass seconds changed from
`7.34917/6.55270/0.70181` to `8.17429/7.37218/0.70059`. This records slower
timing without an observed scientific-metric regression on this benchmark.
The cause is not established: contention/scheduling variation are hypotheses,
not findings. Cumulative asynchronous search/preparation timing is not an
exclusive GPU-time breakdown.

Latest official quota is **89.15/100 GiB used, 10.85 GiB free**. This
read-only status turn authorized no further code change or job; all previous
failures, immutable bindings, quota checks and stopped outcomes remain intact.

## Source126 full CPU/profiles passed; native waiting for physical GPU, 19 October 2026 14:16:49 UTC

The [fresh saved snapshot](../outputs/hpc-results/source126-loss-validation-speed-attempt001/status-news001.json)
at 14:16:49.772386 UTC confirms CPU 29755279's full gate passed, original
repair profile passed and 256-point current-epoch-36 preparation profile
passed with exact scientific receipts in both 128-row chunks. The CPU
allocation record is completed/return code 0. JUnit collected 1,126 cases:
**1,061 passed, 65 CUDA-only skipped, zero failures/errors**, suite duration
187.472s. Skips are 52 event, four checkpoint-replay, three existing native
and six new loss-validation CUDA cases; they remain mandatory in the native
allocation and are not CPU evidence for CUDA behavior.

After-CPU **29755280 completed successfully** (saved LSF log, 15s) and
queued **native 29755832/after-native 29755833**, both PEND. The native
`bjobs -l` query succeeded: 16 CPU slots are reserved on `n-62-18-8`, but
the physical GPU reservation (`ngpus_physical`, one host) is unsatisfied.
The six-hour/80-GB A100 request is valid and has no ineligible wait. The
combined jobs query's return code 255 reflects only aged-out CPU IDs;
saved successful CPU/after-CPU completion and current native PEND are
separately established.

**The GPU has not started.** No source126 native runtime/result, selection,
promotion or training is established. This CPU pass does not establish that
the remaining conservative runtime margin has been cleared. Current official
quota is 88.94/100 GiB used, 11.06 GiB free; every native/retention/runtime/
quota gate remains unchanged, with a fresh promotion check required. This
status check read saved metadata only and made no submissions or code changes.

## Source126 sealed/staged; CPU validation queued, 19 October 2026 12:33:16 UTC

Source126 attempt001 is sealed with **564 authenticated staged members** and
actual **CPU 29755279/after-CPU 29755280 submitted**, both PEND in the
[12:33:16.780303 UTC snapshot](../outputs/hpc-results/source126-loss-validation-speed-attempt001/status-submitted001.json).
CPU budget is one hour/16 slots. **No numerical source126 pass, GPU job or
training is established.** [Preparation](../outputs/hpc-submissions/source126-loss-validation-speed-attempt001-preparation.json)
and [suite](../outputs/hpc-submissions/source126-loss-validation-speed-attempt001.json)
bind archive `53d5fb13f4ad615afb6f0e1601be52325ef8379596951cf1dee694d2f5114967`,
manifest `7b7006dc1465f3167a8a30c828f69225011fb994ca1fee3d5ad8f7c0daedd0ec`
and suite SHA256 `3a0de8659ff38f4f9f3e9bb4512bd74268fd4883709cbdb9fe9a92c8cdc8120c`.
[Upload/staging](../outputs/hpc-submissions/source126-loss-validation-speed-attempt001-upload-stage-output.json)
and [actual queue output](../outputs/hpc-submissions/source126-loss-validation-speed-attempt001-queue-output.json)
retain the receipts. Six exact overlays cover two loss modules, controller
and three test modules; static reviews passed, with the protocol unchanged.

Its scope is removing repeated accepted-target content verification **within
one public loss invocation**, where prepared-identity checks and original
valid-target NLL currently verify the same target content. The freshly
validated records may feed private arithmetic helpers only during that call.
This is not a persistent/object-identity cache, caller-trusted preparation
shortcut or public validation-skip option.

Every public loss call must validate current decoded/target/prepared content
afresh and reject mutations. External mapper guards, full preparation identity,
target/source/coverage provenance, live-loss mutation rejection and native
checks remain intact. Reuse must preserve the same tensor graph, arithmetic,
scalar/components, normalizations and atom/charge/bond/parameter VJPs; no
coefficient, search budget, reduction order, precision or scientific gate is
changed. Existing source125 failures and all sealed releases remain immutable.

Declared additions are 33 base-loss, 22 failure-loss CPU/CUDA and 56
predecessor/metadata fixtures, not executed pass counts. Required native CUDA
cases cannot silently skip; the original 104 event fixtures and CUDA/precheck
commands are unchanged. The byte-pinned source124 attempt002 oracle remains.
Immediate stopped source125 proof and fresh terminal/no-successor guards are
bound. The [predecessor read](../outputs/hpc-results/source125-failure-speed-attempt001/status-pre126-001.json)
records all four source125 jobs DONE and no training. Current official quota
is **88.93/100 GiB used, 11.07 GiB free**, not reserved. Conditional next
stages remain a six-hour/16-slot A100 native comparison, then at most
24h/23h-internal serial training only for an eligible fully passing winner
and fresh quota.

Mandatory allocated-HPC fixtures must cover repeated public calls, mutations
between calls, malformed/tampered targets/receipts and exact scalar, decoder
head and parameter VJP comparisons to the sealed oracle. Only then may a new
same-epoch-36 matched native comparison test the remaining conservative
runtime gap at the unchanged gate. Static edits or avoiding duplicate CPU
validation do not establish a speedup or passing continuation. Exact delta,
release bindings and CPU receipts are recorded above; GPU/numerical outcomes
remain pending.

## Native speed comparison completed; conservative fit still blocks promotion, 19 October 2026 12:21:21 UTC

The [fresh saved outcome](../outputs/hpc-results/source125-failure-speed-attempt001/status-news001.json)
confirms **GPU 29754860 DONE**, executed 13:53:45–14:10:23 DTU-local
(998s), and **after-native 29754861 DONE**. All 104 event fixtures, 13 parity
prechecks and all 18 original native gates passed for all three arms. Selection
has no winner and no training was submitted: supplementary readiness remains
mandatory, and a completed benchmark/controller is not adopted training.

| Arm | Mean epoch projection | Conservative epoch projection | Supplementary outcome |
| --- | ---: | ---: | --- |
| `legacy3` | 10.7276h | 11.67746h | All passed; control only |
| `fallback` | 13.5153h | 14.6271h | Runtime passed; raw-prior-validity retention failed (149→131/256) |
| `bond6` | 13.63167983h | 15.15375995h | Only `conservative_complete_epoch_fits` failed; every other check passed |

Bond6's mean projection is about 28.1% lower than source124's 18.96093496h,
while its fixed-panel validity after eight disposable updates is still
149→172/256. This is native benchmark timing/health evidence, not a measured
training epoch or a long-run chemistry result. The conservative value is
**54,553.5358335s**, exceeding the unchanged **54,000s** protected-estimate
limit by **553.5358335s**. The slowest observed update is 8.17991481s.
Roughly a further 1% reduction in that estimate is needed; its attainment is
not guaranteed by a proposed optimization. The actual gate remains
`1.5 * T + 1800 < 82800`, with no tolerance or margin waiver.

Latest quota is 88.84/100 GiB used, 11.16 GiB free. The user's existing
authorization covers further semantics-preserving execution work to clear
the gates. Removing remaining duplicate target validation within one
public loss invocation is being investigated; all external-mapper/live-loss
mutation, provenance and native checks must remain. **No new source release,
test submission, training job or passing outcome is established for that
next attempt.** Seal and review it separately, rerun required validation,
then promote only a fully passing eligible winner with fresh quota.

## Resource queue wait; tentative scheduler estimate, 19 October 2026 11:42:06 UTC

The [saved detailed queue snapshot](../outputs/hpc-results/source125-failure-speed-attempt001/status-queued001.json)
records native **29754860 PEND**, `Job slot limit reached` on five hosts,
with 14 of the required 16 CPU slots reserved on `n-62-18-11`. The gpua100/
80-GB GPU/six-hour request is valid; this is a normal resource wait, not an
argument or dependency failure. Scheduler estimate: **no earlier than
19 October 19:33:21 DTU-local**, subject to change and not guaranteed.
Callback 29754861 remains PEND; CPU/after-CPU are DONE. Native outcomes and
training remain pending; no additional diagnostic job is needed during this
queue wait.

## Full CPU/profile passed; matched native GPU queued, 19 October 2026 11:40:58 UTC

**CPU 29754828 DONE** passed the full CPU gate, original repair profile and
new matched preparation profile; **after-CPU 29754829 DONE** queued **A100
native 29754860/after-native 29754861**, both PEND in the
[saved full CPU/profile/queue evidence](../outputs/hpc-results/source125-failure-speed-attempt001/status-cpu004.json).
Pytest remains 956 passed/59 CUDA-only skips/zero failures/errors, with the
three recorded timing-property warnings. Full native outcomes and new
training remain pending. The native allocation is one 80 GB A100/16 CPU
slots/six hours for the three matched arms; only a fully passing winner and
fresh quota can enter serial continuation from epoch 36 toward 100 total.

The new CPU profile used **256 frozen outputs from the current epoch-36
decoder**, in two 128-row chunks with six original failed rows altogether.
Both chunks had exactly equal complete preparation receipts/identities:

| Chunk | Sealed old preparation, profiled serial wall | Optimized preparation, profiled serial wall | Full preparation parity |
| --- | ---: | ---: | --- |
| 0–127 | 2.973036993s | 0.607114519s | Exact |
| 128–255 | 5.214458580s | 1.039967208s | Exact |

The aggregate is about **4.97× faster serial preparation under cProfile**,
not a GPU training/update/epoch speedup. All 192 candidate checks per failed
search, coverage and witness cost are unchanged. Profiled extra search per
failed row was about 1.17–1.24s old versus 0.15–0.17s fast. Optimized phase
walls were snapshot/source-validation/mapped-search/assembly-and-identity
`0.0167/0.1517/0.3321/0.1058s` for chunk 0 and
`0.0155/0.1516/0.7284/0.1436s` for chunk 1. These remain serial coordinator
phases, not additive exposed GPU critical-path costs. Small fixture timing
properties also reported roughly 4.95–5.04× (about 0.56→0.11s); they do not
replace the current-checkpoint profile or full native runtime gates.

Latest official quota is **88.84/100 GiB used, 11.16 GiB free**, not reserved
capacity. Scientific settings, gradient/retention/runtime/quota gates and
the original failed source124 outcome are unchanged. No production GPU
speed result, native pass or new-objective training is established yet.

## Pytest prefix passed; complete CPU/profile gate pending, 19 October 2026 11:38:38 UTC

Source125 attempt001 pytest completed **956 passed, zero failures/errors,
59 CUDA-only skips and three warnings in 115.64s**. All 63 new tests passed,
including full search order/check/cost/witness parity, sealed-oracle
preparation identity/loss/VJP parity and mapper-mutation rejection. The
[saved CPU prefix](../outputs/hpc-results/source125-failure-speed-attempt001/status-cpu002.json)
retains JUnit/log evidence. The three warnings concern pytest `record_property`
timing properties under xunit2 schema; they did not fail or waive a check.

At this snapshot **CPU 29754828 is still RUN**. The old CPU repair profile,
new matched current-epoch-36 preparation profile and complete CPU gate remain
pending; the pytest prefix is not a full CPU/profile pass or a measured
production speedup. GPU and training remain unsubmitted. All numerical work
is allocated HPC; local documentation only reads saved metadata.

## SSH restored; source125 CPU submitted, 19 October 2026

After the user reported restored SSH access, the
[upload/staging retry](../outputs/hpc-submissions/source125-failure-speed-attempt001-upload-stage-retry001-output.json)
succeeded and authenticated **561 staged members** of unchanged source125
attempt001. Actual **CPU 29754828/after-CPU 29754829 were submitted**
([queue output](../outputs/hpc-submissions/source125-failure-speed-attempt001-queue-output.json));
this is not a numerical pass. The CPU stage requests one hour/16 CPU slots,
no GPU. Final static review passed the exact eight overlays, controller,
profiler and unchanged gate boundaries. The initial timeout below remains
preserved as a separate transport failure.

All numerical fixtures, preparation-profile attribution and speed results
remain pending. **GPU and training are unsubmitted.** If CPU/profile and
origin/health checks pass, the conditional native stage requests one 80 GB
A100/16 CPU slots/six hours for the three matched arms; serial training stays
at most 24h/23h internal and proceeds only for a fully passing eligible winner
with fresh quota. The original release hashes, loss/search/ratio/state and
every native/adoption gate are unchanged.

## Initial source125 sealing and transport timeout, 19 October 2026

At the first transfer attempt the execution-only release was sealed locally,
**not uploaded or submitted**.
[Preparation](../outputs/hpc-submissions/source125-failure-speed-attempt001-preparation.json)
and [suite](../outputs/hpc-submissions/source125-failure-speed-attempt001.json)
bind archive `4e82615bef55c4a11de04f1401022e3111336ccbfc671bcd2afb64d1ee7d5620`,
manifest `942a35eb1c84033de869e5d9b1d7f2752f45d5ff3232337dcb56467a474c80c2`
and suite SHA256 `49a7e8f03a5c199f8834a41b93a2f7e5f1af89ff89d3b01abaea495da4e96209`.
Exactly eight overlays comprise three CPU execution modules, the sealed
preparation-oracle loader, controller, profiler and two test modules. All
configuration bytes, loss/ratio/search settings, epoch-36 lineage,
epoch-35 reference and scientific/native gates are unchanged.

The [first transfer/staging output](../outputs/hpc-submissions/source125-failure-speed-attempt001-upload-stage-output.json)
records SSH return code 255, `Connection timed out`, before login/control
creation. No upload, remote staging or job occurred at that failed attempt.
Fresh SSH readiness was requested before the later successful retry above.
Controller/profile review was pending at that stage. Preserve this transport failure
separately from scientific checks and do not modify the sealed archive.

The new **46 search and 17 preparation fixtures** are mandatory before the
allocated CPU attribution profile. They are declared cases, not passes.
The existing 104 GPU event fixtures and all later native/quota/adoption gates
are unchanged. Numerical validation, measured speed improvement and a training
continuation remain unestablished; no local numerical test was run.

## Authorized scope and current status, 19 October 2026

The user asked why the roughly 2% of molecules that still fail the original
repair search add substantial cost, and authorized execution speedups,
allocated DTU HPC benchmarks and conditional continuation only after all
existing checks pass. **22/1,024 = 2.15% is a molecule fraction, not a timing
fraction.** Source125 implementation is sealed locally as attempt001 above;
controller/profile static review and authenticated transfer have now passed.
Allocated validation remains pending; actual CPU submissions are recorded
above and no measured new speedup is established. All numerical references, profiling,
parity tests and benchmarks run on allocated DTU HPC. Local work is editing,
static review, packaging and saved-file metadata analysis only.

Preserve [source124 attempt002's completed outcome](../outputs/hpc-results/source124-deterministic-events-attempt002/status-news001.json)
and every earlier source123/source124 failure. The deterministic event fix
passed allocated fixtures/native gradient checks, but neither new objective
was eligible: fallback failed raw-prior-validity retention and conservative
epoch fit; bond6 failed original complete-epoch fit. Bond6's 12 complete valid
and 10 invalid partial witnesses and its eight-update 172/256 raw-valid
observation are short benchmark results, not a trained endpoint. Faster
execution alone does not establish retention or authorize loosening a gate.
See [docs65](65_deterministic_repair_event_recovery.md).

## Failed-molecule fraction and measured cost are different

The 22 failed native-panel rows are only a small subset of 1,024 points,
but failure preparation authenticates all 128 rows in each microbatch. The
sealed bond6 mapped path hashed original raw labels/probabilities before
dispatch and again while assembling rows; accepted targets also received
duplicate validation. Cost therefore is not restricted to the 22 rows that
need new fallback/search guidance. The additional preparation, differentiable
loss and synchronization work must be measured rather than inferred from
the failed-molecule percentage.

The strongest saved timing comparison is the **first complete update from
the exact same starting state**, before candidate trajectories diverge:

| Existing source124 candidate | First update wall | Original search checks | Extra checks |
| --- | ---: | ---: | ---: |
| `legacy3` | 5.2020s | 20,832 | 0 |
| `fallback` | 8.3126s | 20,832 | 0 |
| `bond6` | 11.2270s | 20,832 | 1,920 |

Bond6's 1,920 extra checks are ten failed prior rows × 192; this first-update
cohort is separate from the fixed 1,024-point panel's 22 failures. Fallback's
extra wall occurs without any extra candidate search, demonstrating broader
failure-guidance work. These timings do not by themselves assign that cost to
one function. Across eight disposable updates, mean walls were
5.4931/7.2601/10.2579s for legacy/fallback/bond6 respectively, but their later
weights/targets differ, so those means are not an exact frozen-input cost
comparison. The [saved source124 outcome](../outputs/hpc-results/source124-deterministic-events-attempt002/status-news001.json)
remains authoritative; these are old measured costs, not source125 results.

In the existing streamed trainer, `search_seconds` sums original teacher receipt
elapsed time **and** additional `FailureTeacherBatch.preparation_seconds`.
The latter wraps failure preparation after the original teacher and may
include extra search, diagnostics, target construction and verification.
These cumulative CPU preparation durations are not automatically an exposed
critical-path fraction in the asynchronous update.

`prepass_seconds` currently covers the detached encode/decode/statistics and
teacher-launch loop. In overlap mode, queue waits happen later while consuming
prepared batches. It does not measure the whole teacher lifetime or the
complete training update. The separate native-panel search diagnostics are
also outside synchronized production-update timing. Do not divide one of
these fields by an unrelated duration and infer where most wall time goes.

The allocated profile must separate and label:

| Phase | Required measurement/interpretation |
| --- | --- |
| Original teacher | Candidate checks, CPU wall/work, cache scope and ordered results |
| Additional search/preparation | Extra candidate checks, diagnosis/native checks, hashing/verification and target construction |
| Queue/pipeline | Submission/copy overhead and exposed waits; no double-counting overlapped work as serial wall |
| Differentiable guidance | Forward/backward of original NLL, extra valid/partial NLL and local/cut/nonempty terms |
| Other update work | Encoder/decoder, composition/occupancy, gradient probes, clipping and optimizer |
| Complete update | Synchronized wall time including all required work, warmup and measurement boundaries |

Detailed probes must declare their overhead and remain distinct from the
production timing used by runtime gates. Compare equivalent measured
repetitions, report mean and slowest complete-update extrapolations, and retain
the immutable raw timing records. A small search share does not rule out a
larger differentiable-loss or synchronization cost.

## Implemented source125 execution changes; numerical validation pending

The bond-only extra search now reuses existing `BondOnlyDiagnostic` and
`BondSearchContext` helpers for fixed atom/charge native templates,
probability-regret tables and graph-state work. It preserves the original
candidate order, six-edit/32-check/beam-four budget, seen-state equivalence,
tie/witness costs and receipts. The unchanged old implementation remains
`search_progress_repair_reference`; atom-edit diagnostics retain that oracle
path. Full public validation remains at entry and on the retained witness.
No search depth, candidate allowance or scientific acceptance rule was reduced.

Failure preparation authenticates all rows once into private invocation-owned
views, retaining a receipt digest/candidate/target guard across an external
mapper. Full mutable accepted-target validation still occurs after the mapper,
and the live loss still independently validates complete prepared identity and
targets. Reusing the target records from that same authenticated invocation
avoids duplicate identity-time validation without introducing a caller-trusted
cache or removing mutation rejection. GPU loss arithmetic is unchanged from
source124 attempt002.

Optional preparation phases (`snapshot`, `source_validation`,
`mapped_progress`, `assembly_and_identity`) are coordinator timings outside
scientific identity. They are not additive exposed GPU critical-path costs.
The reference execution backend now explicitly uses the original progress
oracle, while optimized workers use the fast path, so existing native
execution-parity checks compare implementations rather than two calls to the
same optimized code. Inline comments/docstrings state these boundaries;
allocated parity is still required before claiming equivalent execution.

The new allocated CPU profile
[`benchmark_failure_preparation.py`](../scripts/benchmark_failure_preparation.py)
authenticates the **current epoch-36** checkpoint and compares 256 matched
frozen decoder outputs in 128-row chunks, using identical original targets.
It loads the original preparation module from its exact sealed source124
release, compares complete preparation receipt/identity, and records old/new
serial `cProfile` plus phase timings. Its 16-slot/four-main-thread allocation
does not turn serial extra-search profiling into a 12-worker GPU epoch
projection. Profile overhead is explicit; only the full native synchronized
update benchmark may establish promotion runtime.

## Scientific contract that cannot change

Keep the authenticated completed epoch-36 model/checkpoint
`3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f`,
AdamW state, native RNG and existing fresh-prior RNG. Preserve the frozen
epoch-35 reference and original repair-ramp lineage. Never reseed or advance
the eventual training state through disposable benchmark updates.

Retain joint encoder/decoder training, FP32 model parameters, 1:2 ratio,
256 real/512 prior points, microbatch 128, 12 worker processes/four main
threads within 16 CPU slots, all coefficients and 0.25/0.75 branch means,
composition/occupancy/reconstruction/KL semantics and the scoped FP64 event
normalization. Keep six-edit extra bond search, beam 4/checks-per-depth 32,
original three-edit teacher and unchanged candidate/check budgets. Production
atom edits remain off; partial targets remain invalid and complete valid
witnesses remain independently checked.

An execution optimization must preserve ordered candidate generation,
deduplication, round-robin traversal, checked-count accounting, frontier
ranking, deterministic ties, earliest equal-cost witness behavior and the
valid-versus-partial selection rule. It must not trade fewer checks, a
different stopping rule, reordered probabilities, altered loss normalization
or weaker native conversion for speed. Probability floors remain detached
ranking-only; differentiable terms stay unfloored. Cached/reused work must
retain exact raw/probability/target provenance and mutation rejection, with
physical work saved distinguished from the logical candidate count.

## Proposed validation and staged release

Before submission, seal a new source including uncommitted changes, explicit
delta from source124 attempt002, configuration, exact benchmark grid,
resources, parent/reference/data identities and fixture commands. This
document does not invent candidate IDs, test counts or job receipts before
that release exists. Preserve the old stopped controller and all original
selection/promotion evidence; a retry must authenticate its exact origin,
terminal jobs and lack of a competing successor.

Allocated CPU fixtures first compare optimized search to the sealed reference
on identical supported/unsupported, already-valid, censored, complete/partial,
tie, duplicate, exhausted-frontier, maximum-budget and atom-disabled cases.
Require identical witness labels/identity/cost, status, diagnostics, ordered
logical checks and target/provenance semantics, plus unchanged failed-input
behavior. Retain baseline NLL/normalization parity and all existing controller,
resume, handoff and error-rejection tests. No local numerical smoke tests are
substitutes for this stage.

Allocated GPU validation retains strict deterministic algorithms and the
mandatory CUDA flag/prechecks. Compare forward values, head/parameter VJPs,
fixed-preparation finite differences and repeatability at the existing
FP32/FP64 tolerances, including empty and maximum-cut/extreme-logit cases.
No warning-only determinism or larger tolerance is authorized. Any altered
reduction grouping needs explicit measured parity before a speed claim.

Native matched benchmarks then reset from the same epoch-36 weights,
optimizer/RNG/context and declared points. Keep the existing 1,024-prior panel,
fixed 256-point retention panel and eight complete disposable updates where
required by the sealed plan. Authenticate all original 18 native and
supplementary gates, 13 parity observations/eight timings and actual
allocation/run/source bindings. Report search/gradient coverage with all and
eligible denominators; zero/unsupported/error cases remain visible. A
profiling run is not an adopted checkpoint or a complete epoch measurement.

## Conditional continuation remains fail-closed

The complete-epoch safety condition is unchanged:
`1.5 * T + 1800 < 82800` seconds, including the conservative slower-update
estimate where required. Retain raw-prior-validity, composition/sulfur/size/
nonempty/reconstruction, gradient/native/parity/memory/provenance and every
other readiness check. An old failed result cannot be approved retrospectively
because a later implementation is faster. If no new candidate passes every
gate, record the stopped outcome without training.

Only a passing eligible new-objective candidate may proceed through the
unchanged selection rule and independent fresh official-quota check. The
latest saved quota before this work is 88.70/100 GiB used, 11.30 GiB free;
prior [verified offload](50_hpc_space_cleanup.md) is preserved, and space is
not reserved. Use unique campaigns, no concurrent training, allocations at
most 24h/23h internal and at most 100 **total** epochs, preserving all original
and fixed-handoff epoch health/identity/quota checks. Final evaluation requires
the authenticated absolute epoch-100 endpoint; freeze a resulting decoder
before latent optimization. Actual submission, pass, training-start and
completion claims require their own receipts in the ledger.
