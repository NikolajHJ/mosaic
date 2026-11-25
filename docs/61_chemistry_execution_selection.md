# Gated selection of a faster chemistry-continuation execution path

## Scope and current status

The user authorized implementing and benchmarking CPU/GPU overlap, worker count
and microbatch changes on 17 October 2026. Keep 1:3 if a healthy candidate
projects below ten hours per real epoch; otherwise test a declared 1:2 fallback.
Automatically promote the selected passing candidate only after all native,
identity, completion and quota checks.

**Selected continuation actually queued, 18 October 2026 12:14 DTU-local:**
the passed additive recovery submitted **GPU segment 001, job 29749078 PEND** and
the original **callback 29749079 PEND** on `ended(29749078)` at 12:13:40 and
12:13:42 respectively. Promotion-time official quota passed at 89.43/100 GiB
used, 10.57 GiB free versus 10.125 GiB required. Segment 001 requests one 80 GB
A100, 16 CPU slots/64 GB and 24h allocated /
23h internal. It restores the original authenticated epoch 35 parent, AdamW
and native RNG lineage, creates the separate fresh-prior RNG and does not use
any disposable probe as a training checkpoint. **Training has not started;
no epoch 36 or later checkpoint is established.**

The selected `r2_async_w12_m128_h24` protocol remains 1:2 (256 real/512 prior),
with 12 repair workers/four main threads, microbatch 128, unchanged 0.25/0.75
repair means, prior-mean occupancy anchor and 1.5 prior-count population weight.
Continue serially toward 100 total epochs only while native/health/identity/
quota/completion checks pass; final evaluation remains gated by the authentic
epoch 100 endpoint. CPU 29748998 passed 68 metadata fixtures in 3.19s plus actual
saved-evidence authentication, permitting only the five exact hash-bound
disposable native resume probes. The original source121 scientific code,
native gates, quota-stopped decision/intents and both earlier readiness
failures are unchanged. The 12:14 scheduler read records a host job-slot limit
with 16 slots reserved on `n-62-18-9`; the callback awaits its dependency.
[Verified submission](../outputs/hpc-results/source121a002-quota-promotion-attempt003/local-verification.json),
[actual segment receipt](../outputs/hpc-results/source121a002-quota-promotion-attempt003/receipts/segment-001-receipt.json),
[passed readiness](../outputs/hpc-results/source121a002-quota-readiness-attempt003/local-verification.json),
[verified storage offload](../outputs/hpc-results/verified-capsule-offload-20261018-001/local-verification.json)
and [ledger](05_experiment_ledger.md).

**Earlier space recovery; additive promotion retry pending, 18 October 2026,
superseded by the actual submission above:** the
obsolete Python 3.12 environment was removed with active-input audits and
fresh unchanged epoch 35/original 89 checkpoint hashes. Official quota during
the cleanup interval changed92.01 to 89.72/100 GiB used;10.28 GiB free is a
snapshot, with the unchanged 10.125 GiB initial minimum checked again before
promotion. A separate historical evidence gzip job 29748954 is RUN; its
completion is not claimed. The original quota-stopped decision/intents remain
immutable while a new metadata-only recovery authenticates the already
selected source121 1:2 winner. The first allocated readiness 29748953 passed 57
fixtures, then rejected the aged-out scheduler-message interface. Retry 29748962
passed 59 fixtures and authenticated that terminal proof, then stopped at an
overbroad any-`.pt` guard: each of the five native benchmarks intentionally
retains a hash-bound disposable resume probe. A further metadata-only recovery
will permit only those five exact authenticated probes while retaining every
real training/output/duplicate-submission guard. Both failed readiness records
remain preserved; no scientific gate changes.
**No training has been submitted or started at this update.** The selected
1:2 protocol, epoch 35 parent,100-total-epoch maximum and 24h allocation/23h
internal serial segments remain unchanged.
[Verified cleanup](../outputs/hpc-results/unused-boltz312-cleanup-20261018-001/verification.json),
[storage details](50_hpc_space_cleanup.md) and [ledger](05_experiment_ledger.md).

**Native selection complete; training blocked by quota, 18 October 2026
11:36 DTU-local:** GPU benchmark 29747654 completed 00:43:01–01:11:42. All four
CUDA fixtures passed. Each 1:3 variant failed only the unchanged
`complete_epoch_fits` safety gate; its other 17 native gates were true. The
generic reader rejection mentions source/parent/plan/job/completion differences,
but the saved raw gates establish runtime fit as the failure, not an identity
or model-instability finding.

Both 1:2 overlap variants passed all 18 native gates. The declared fallback
selected **`r2_async_w12_m128_h24`**,12 workers/four main threads/microbatch 128,
from eight timed disposable updates. Its native projection is41,509.414s/epoch;
the conservative selection projection is44,344.675s (about 12.32h/epoch), versus
45,288.641s for microbatch 256. This is above the ten-hour preference target and
is permitted by the declared1:2 fallback; it is not a measured full epoch or
a continuation checkpoint. Repair means remain0.25/0.75, occupancy anchoring
remains prior-mean and composition prior counts use weight 1.5.

Promotion29747655 then stopped at the hard work3 quota gate with
`Insufficient work3 quota for serial chemistry phase`. Fresh quota is91.50/
100GiB used,8.50GiB free against the unchanged 10.125GiB initial training
minimum. **No training was queued or started; the epoch 35 parent is unchanged.**
No job, setting or gate was changed in this status turn.
[Verified capsule004](../outputs/hpc-results/source121a002-first-suite-004/local-verification.json),
[selection](../outputs/hpc-results/source121a002-first-suite-004/runs/source121a002-chemistry-16core-24h-attempt002/selection.json),
[stopped promotion](../outputs/hpc-results/source121a002-first-suite-004/runs/source121a002-chemistry-16core-24h-attempt002/variants/r2_async_w12_m128_h24/advance-after-benchmark.json)
and [ledger](05_experiment_ledger.md).

**Latest queue status, 18 October 2026 00:39 DTU-local:** source121 attempt002
**GPUbenchmark29747654 PEND / promotion29747655 PEND**. LSF reports insufficient
processor units/affinity resources at the eligible GPU host, with11 of16 slots
reserved; `done(29747653)` is satisfied. Its current forecast is no sooner than
02:38:14 today, without a guaranteed start. No native timing, winner, selected
ratio or training result exists. The completed CPU715-pass/four-CUDA-skip gate
and CPU repair profile remain historical passes; the four CUDA checks remain
mandatory. This is a resource wait, not invalid arguments or a new model
failure. No job, setting or check changed in this read-only status turn.
[Verified capsule003](../outputs/hpc-results/source121a002-first-suite-003/local-verification.json)
and [ledger](05_experiment_ledger.md).

**Reduced suite CPU/profile passed, 17 October 2026 22:57 DTU-local:**
source121 attempt002 CPU29747653 DONE:715 passed/four CUDA-only skipped
in68.16s, plus a passed frozen-parent CPU repair profiler. Six-hour GPUbenchmark29747654 and
promotion29747655 PEND on their declared dependencies. The suite uses16 CPU
slots/64GB,12 repair workers/four main threads and the five variants below;
future selected segments are24h allocated/23h internal. No native speed,
winner, selected ratio or training result exists. The original source118
benchmark/callback are confirmed intentionally retired before native execution.
Source121 attempt001 failed only new fixture `str`/`Path` setup (45 failed,
670 passed/four CUDA skips); attempt002 changes that fixture conversion only,
preserving production/settings/assertions. Its blocked predecessors were
confirmed retired. The CPU diagnostic profile is not a native-update/epoch
benchmark or a changed training ratio. Last9.85GiB free falls below the
unchanged10.125GiB initial training reserve; fresh quota remains a hard gate.
[Verified current evidence](../outputs/hpc-results/source121a002-first-suite-002/local-verification.json),
[preparation](../outputs/hpc-submissions/source121a002-preparation.json)
and [ledger](05_experiment_ledger.md).

**Historical planning snapshot, 17 October 2026 22:39 DTU-local, superseded by
the actual status above:**
the user requested at most24h allocations and fewer benchmark CPUs. The new
grid will hold12 search workers/four main Torch threads within16 slots and
defer20/28-worker cases. It retains1:3 sync/micro128 control and1:3 async
micro128/256, followed only if needed by1:2 async micro128/256: **five planned
variants**. All eight timed updates, native/parity/coverage/retention/identity/
quota checks and the1:3-first selection policy remain mandatory. Future selected
training segments become23h internal/24h allocated; complete-epoch fit is
recomputed under that cap, not waived. No loss or ratio-normalization change is
implied. Source118 remained PEND at22:39; new source/retirement/submission
receipts are pending. No speed, selected ratio or training start is claimed.
The13-variant/32-slot and72h records below remain historical source118 facts.

**Allocation audit, 17 October 2026 22:30 DTU-local:** the pending source118
benchmark has a6h allocation/5.5h internal cap,32 CPU slots/128GB host memory
and one80GB A100. Only a future passing selected continuation requests72h/71h
per serial training segment. The inspected eligible host has32 physical cores,
about1006.8GiB RAM and two80GB GPUs; a one-GPU/32-core request fits its static
inventory. Combined GPU affinity is `aff=no`; core affinity does not impose a
one-GPU/16-core ceiling. Present host capacity is occupied/reserved. LSF's
current forecast says not before20 October11:10:17 DTU-local, which is not a
guarantee and may change. Live queue maximum72h conflicts with the public DTU
GPU page's older24h statement. No arguments, jobs or checks changed.
[Read-only scheduler/host proof](../outputs/hpc-results/lsf-argument-audit-20261017/independent-readonly/scheduler-host-audit.json)
and [ledger policy references](05_experiment_ledger.md).

**Latest source118 status, 17 October 2026 22:14 DTU-local:** GPU29745548 and
promotion29745549 remain PEND; no native candidate timing, selected ratio or
chemistry training exists yet. Its completed CPU/profile pass remains valid.
The separate queued source119 Boltz/DeFoG comparison uses the original frozen
epoch-89 VAE, not this newer four-round triangle parent at epoch35. No VAE
settings/submissions changed and no ETA is established.
[Verified capsule006](../outputs/hpc-results/source118-first-suite-006/local-verification.json),
40 members /39 verified files, exported `2026-10-17T20:14:31.229014+00:00`,
SHA256 `953fb2c90d0ca0e2c61d2ad1523c922b7dae7581bae9df598b75966d8202c8f4`.

**Earlier source118 status, 17 October 2026 21:03 DTU-local:** GPU29745548 and
promotion29745549 remain PEND; no native timing, selected ratio/winner or training
result exists. The persisted CPU/profile pass is unchanged despite the completed
CPU job aging out of `bjobs`. The coordinator's eligible-host read is
`closed_Full` (MAX32/NJOBS32/RUN24/RSV8): capacity is occupied/reserved, not a
physical GPU count or proof of an impossible request. No ETA or VAE changes.
[Verified capsule005](../outputs/hpc-results/source118-first-suite-005/local-verification.json),
40 members /39 verified files, exported `2026-10-17T19:03:04.422836+00:00`, SHA256
`52fa35b6c3aa1cbe6b05a0308277ce7f88764791ad7f732d3ec90d74dbce5d6b`.

**Earlier source118 status, 17 October 2026 17:32 DTU-local:** GPU29745548
remains PEND for processor-unit/GPU-affinity availability on one host; promotion
29745549 remains dependency-pending. Native candidates have not run, with no
timing, winner, adopted ratio or training start. CPU29745547 is no longer in
`bjobs`, whose combined read returned255 only for that missing completed job;
the persisted passing CPU/profile gate and unique authenticated allocation
(`completed`, returncode0) remain valid. The existing aged-out completion path
was independently checked. This is a resource wait, not loss of the CPU pass
or evidence of a new model/validation failure. No changes/submissions occurred.
[Verified capsule004](../outputs/hpc-results/source118-first-suite-004/local-verification.json),
40 members /39 verified files, exported `2026-10-17T15:32:44.744263+00:00`, SHA256
`7566cc8dddd05133e4978002a61410d7c79b992a9ee17e8a518b35d3c6c61013`.

**Earlier CPU completion, 17 October 2026 13:41 DTU-local:** **CPU29745547
DONE / complete CPU gate passed**, with669 tests passed/four CUDA-only skipped
in70.48s and profiler returncode0. Profile SHA256:
`cee2f508efe30ce9a36ab5ac8b18ad99188a5ce47c8b34a0c324b80c678e9ed3`.
GPUbenchmark29745548/promotion29745549 remain PEND. All four CUDA checks stay
mandatory in the GPU stage. No native timing, speed/winner, adopted1:2 protocol
or training start is established.
[Verified capsule003](../outputs/hpc-results/source118-first-suite-003/local-verification.json),
40 members /39 verified files, exported `2026-10-17T11:41:07.016812+00:00`, SHA256
`730f58d2eb0b49a98590c2f22e07b7714531a82bf232d7046762a6ea1f191b6f`,
preserves the passing CPU/profile gate and pending native status.

**Capacity update, 17 October13:47 DTU-local:** bounded removal of five redundant
remote transfer archives (all byte-identical local copies/extractions retained)
left10.31 GiB officially free, about0.185 GiB above the unchanged10.125 GiB
initial guard. This is not reserved capacity; promotion must recheck quota.
GPU29745548/callback29745549 remain PEND, with no native timing/winner/training.
[Maintenance receipt](../outputs/hpc-results/verified-capsule-offload-20261017-002/completion.json)
and [docs50](50_hpc_space_cleanup.md) retain exact guards, paths/hashes and the
earlier zero-unlink PATH failure. No scientific setting or gate changed.

**Historical fixture-only snapshot, 17 October 2026 13:38 DTU-local:**
[Verified capsule002](../outputs/hpc-results/source118-first-suite-002/local-verification.json),
38 members /37 verified files, exported `2026-10-17T11:38:51.565100+00:00`, SHA256
`0fc9c43121e01cb938863491e7f592089c0535200e1415362adf450f03055727`,
preserves that result; source117's earlier failure remains history.

**Historical source118 submission, 17 October 2026 13:36 DTU-local:**
[Verified capsule001](../outputs/hpc-results/source118-first-suite-001/local-verification.json)
authenticates source/suite/all 13 plans/parent/three receipts, with 38 members /
37 verified files, exported `2026-10-17T11:36:46.589255+00:00`, SHA256
`bd230552c9a696887eba2d7ae804d5d8c064c89828bd4832b2c19be19d55bc20`.

[Source118 preparation](../outputs/hpc-submissions/source118-preparation.json)
binds archive `2664348dc695879e5eb5e3546dd70b03577bc02a31b4e0da102e1efae3a931b9`,
manifest `1d3126e43b1b4be4309130567480116c2c0432c6d33336d571e65207a688f0a4`
and suite `24e38fbb8f592337f91799b0bc3af53f83e1845a9e6b38f9416455e299a024af`.
Only three fixture lines add missing constructor-equivalent directory setup;
all production files, model configurations, assertions and checks remain
byte-identical to source117. All 532 staged members were authenticated. Current
CPU resources remain 16 slots/64GiB/2h; native GPU remains 32 slots/128GiB/6h.
Source117's failed CPU gate remains preserved; blocked GPU525/callback526 were
confirmed owner-canceled EXIT 13:35:44 before execution. No failed check was
waived. Source114 remains untouched. New roots use
`source118-chemistry-speed-selection-attempt001` (release basename
`source-118-chemistry-speed-selection-attempt001`).

**Historical source117 failure, 17 October 2026 13:33 DTU-local:** CPU29745524
failed one fixture (668 passed/four skipped,71.31s) before profiler/native work.
The positive shared-provenance fixture used `__new__` without the production
constructor's directory creation, causing `FileNotFoundError` while writing
`authenticated_selection.json.tmp`. This is a fixture setup error; the gate
remains failed, with no native speed result or training start. The
[verified capsule002](../outputs/hpc-results/source117-first-suite-002/local-verification.json)
preserves the failure. Pending GPU29745525/callback29745526 cancellation
requests were accepted13:34:17, with terminal confirmation pending at this
entry. Source118 test-only recovery is in preparation, retaining assertions
and all scientific/configuration checks; no submission/pass is claimed yet.

**Historical source117 submission, 17 October 2026 13:32 DTU-local:** source117
CPU **29745524 RUN** on `n-62-21-80`, GPU benchmark **29745525 PEND** and
promotion **29745526 PEND**. No completed CPU gate, native check, speed result,
winner, adopted1:2 protocol or training start is established.
[Verified capsule001](../outputs/hpc-results/source117-first-suite-001/local-verification.json)
authenticates source/suite, all thirteen plans, parent and three receipts;
38 members /37 verified files, exported `2026-10-17T11:32:28.431041+00:00`,
SHA256 `3866ea627e2b384967eb21bba0e9b42094ffa521a67dad5801c9f1d73957b492`.

The new [preparation](../outputs/hpc-submissions/source117-preparation.json)
binds archive `a29dba5d9ba472eaf4eb46de3bbf51f85f386044e3f7168108781aec02b7a5d3`,
manifest `f914d6d39e9a3e9c4e6bcbe23dfbd435e7ccc291e49ad9bb3d5f042287dafbd1`
and suite `8d14a90b125ff48cf2796f1491e52a1ec8ab4d9c8661822db737ff69ce645d63`.
All532 staged members were authenticated. Relative to source116, only CPU
resource control and matching fixtures change: CPU preflight/profile now uses
16 slots /64GiB for2h, which covers the profiler's12 workers plus four main
threads. The native GPU stage remains32 slots /128GiB /one A10080GB /6h,
with the same13 configurations and scientific checks. New plans bind the new
source/run identities; no gate was waived.

Source116's three still-pending jobs514/515/516 were intentionally retired:
requests accepted13:29:17, terminal EXIT observed13:30:19. They never ran
numerical work. The
[retirement record](../outputs/hpc-results/source117-first-suite-001/source116-retirement.json)
preserves pending-state requests separately from the coordinator's later
terminal observation. Source114 remains untouched. Source117's evidence
collector includes `.out`/`.err` in its original prepared operator, so the
source116 external collector refinement below remains history, not a new117
recovery. All new roots use `source117-chemistry-speed-selection-attempt001`
(release basename `source-117-chemistry-speed-selection-attempt001`).

**Historical source116 submission, 17 October 2026 13:27 DTU-local:** actual jobs
are **CPU 29745514 PEND**, **GPU benchmark 29745515 PEND** (`done(CPU)` dependency)
and **promotion 29745516 PEND** (`ended(benchmark)` dependency). CPU fixtures
have not run; no native pass, speed result, winner, adopted 1:2 protocol or
training start is established. The
[verified capsule001](../outputs/hpc-results/source116-first-suite-001/local-verification.json)
authenticates all three receipts, source, suite, all thirteen candidate plans and
parent, with 34 members / 33 verified files, exported
`2026-10-17T11:27:24.337901+00:00`. Archive SHA256:
`2c047b0a8af5c786e0867bf3b7eddb8da1ac7abeb0f75687422a594b8c198a08`.

The immutable [preparation](../outputs/hpc-submissions/source116-preparation.json)
binds source archive
`7ba17d868bfe794f5714355cb962f34b1610a8349438b841613e8fd1f5f049ff`,
manifest `63d75d0970f04aa15e214662b5b8a622cadaf4271d71701d975ef2b7b4f6fa5a`
and suite `62ae66182c8a6035d6e91330af286e2352aff05192705ac071d8505174ef45f0`.
Staging authenticated 532 members. Fifteen Python files and thirteen bound
configurations change; 446 inherited files and 135 original model-package files
remain byte-identical. `_bond_edits`, `search_atom_preserving_repair` and
`repair_loss` retain identical reference-function ASTs. These byte/static proofs
are not allocated scientific checks.

Source115's CPU/profile pass remains valid history. Its then-pending benchmark
29745445 and callback 29745446 were intentionally canceled and observed EXIT at
13:15:30 DTU-local, preventing competing automatic training. This is an operator
retirement, not a failed scientific gate. Source114's independent optimization
comparison remains untouched. The
[retirement record](../outputs/hpc-results/source116-first-suite-001/source115-retirement.json)
preserves requested cancellations while both were PEND; terminal EXIT was then
observed by the coordinator. New receipts and identities are in the
[ledger](05_experiment_ledger.md).

Source112's failed epoch-fit result is preserved: a short native benchmark
projected 22.4 hours/epoch and failed the old 22-hour internal segment's unchanged
safety formula. Source115 changed only that internal budget to 71 hours; this
new speed suite additionally changes declared execution candidates and, only
if needed, the finite-sample prior ratio. No earlier failure is rewritten as a
pass. See [docs53](53_atom_preserving_triangle_continuation.md) and
[docs58](58_atom_preserving_repair_performance.md).

## Predeclared candidates and measurements

The current source121 suite has **five** candidate configurations:

| Family | Execution candidates |
| --- | --- |
| 1:3 baseline | Synchronous teacher, 12 workers, microbatch 128 |
| 1:3 overlap | CPU/GPU overlap; 12 workers × microbatch 128/256 |
| 1:2 fallback | CPU/GPU overlap; 12 workers × microbatch 128/256 |

The retired source118 suite had thirteen candidates including20/28 workers;
those worker counts are deferred. This is a declared reduction in execution
coverage, not evidence that the unmeasured larger pools are slower.

All candidates share the authenticated epoch 35/update 230860 parent, frozen
reference copy, native model/data/split, real logical batch 256, full loss/gate
settings and source identity. A separate child process restores the parent
model, AdamW and RNG state for each candidate; its updates are disposable and
never become continuation checkpoints. Equivalent execution candidates within
one ratio must see the same prescribed points. A different ratio deliberately
consumes fewer prior RNG draws and is not bitwise-equivalent 1:3 training.

One serial GPU allocation with16 CPU slots /64GiB requested host memory and
one A100 80 GB GPU evaluates the declared families;
it does not create five independent GPU jobs. Each candidate performs eight
timed complete updates. Include data collation, required synchronization,
search waits and model/update work in those walls; record startup/warmup
separately. Missing/failed/timed-out candidates remain recorded and cannot win.
The source121 CPU stage requests2h/16 slots/64GiB. The GPU suite requests6h/
16 slots/64GiB, with5.5h internal execution budget. A selected training candidate
retains16 slots and the declared23h internal/24h allocation limit. Source118's
retired GPU request was32 slots/128GiB with71h/72h future segments; source116
also briefly reserved32 CPU slots before retirement. These are historical
resource controls, not the current source121 request. Limits are not measured
runtimes or ETAs; the unchanged complete-epoch-fit formula must still pass
under the23h internal cap.

Overlap must preserve the same weights throughout an update, cached posterior
and fresh-prior draws, candidate ordering, bounded search/cache decisions,
global composition statistics and the single optimizer step. CPU processes
receive detached candidate data, not CUDA tensors or model weights. Worker
errors must propagate; no hidden serial fallback or substituted target is
allowed. Compare the actual loss, gradients, model, optimizer, RNG and reference
state against the native reference execution before accepting the candidate.

## Explicit fallback loss semantics

At 1:3 the logical update contains 256 real and 768 prior points. At 1:2 it
contains 256 real and 512 prior points. Both retain:

- Reconstruction and KL on real molecules, unchanged coefficients/ramp.
- Repair `0.25 * mean_real + 0.75 * mean_prior`, with failed and already-valid
  attempts zero within the all-attempt branch mean.
- The prior-mean occupancy anchor and its original frozen reference.
- Element/charge population-JS weighting that represents the original 1:3
  mixture: multiply prior count contributions by `3 / prior_ratio`, hence 1.5
  for ratio 2, consistently in statistics and gradient replay.

Fewer prior draws change Monte Carlo variance and RNG consumption, even while
the intended branch/population weighting is retained. This is an explicitly
versioned sampling protocol, not a claim that finite batches or training
trajectories are identical. Fixed logical real-batch size, identity-safe
candidate numbering, point normalization and short-batch ratios remain checked.

## Native gates and deterministic selection

All eighteen native gates remain mandatory, including exact native resume,
reference execution/restoration, finite and nonzero gradients, memory, search
provenance, prior coverage, reconstruction/validity/composition/charge/sulfur/
size/occupancy retention and epoch-fit safety. The thirteen required reference-
parity checks must all be present and true; a partial proof is not approval.
Do not accept a fast failed variant or substitute a fixture for actual native
checks.

Rank healthy candidates conservatively using the larger of the native epoch
projection and
`max(timed_update_seconds) * training_molecules / real_batch_size + 600`.
The maximum guards against selecting on one lucky median; it is not a full
epoch measurement or statistical confidence interval.

1. Evaluate the complete baseline/overlap 1:3 family first.
2. If any healthy 1:3 candidate has conservative projection **strictly below
   36,000 seconds**, select the fastest of those and do not need 1:2 evaluation.
3. Otherwise evaluate the full 1:2 family and choose its fastest healthy member,
   **even if still above ten hours**. Ten hours is a ratio-preference target,
   not permission to waive a gate.
4. Break exact timing ties by canonical candidate ID. No healthy 1:2 candidate
   means stop without training.

The standard-library helper
[`chemistry_speed_selection.py`](../src/latentmol/chemistry_speed_selection.py)
separates this metadata policy from allocated model work. Focused
[`test_chemistry_speed_selection.py`](../tests/test_chemistry_speed_selection.py)
fixtures cover every gate/parity failure, malformed/incomplete timing data,
preference/fallback/tie policy and nonmutation. Their local verification was
AST parsing only; allocated fixture execution is now covered by the source118
CPU result above. Actual native benchmarks and winner promotion remain pending.

## Promotion and evidence

The controller
[`scripts/hpc/chemistry_speed_campaign.py`](../scripts/hpc/chemistry_speed_campaign.py)
must authenticate complete candidate-family observations, winner identity,
native gate/source/plan/parent/allocation bindings and genuine job completion.
Selection is not itself a training start. Retain every candidate's raw gate,
timed records, stdout/stderr and rejection, plus exact selection rationale and
selected configuration hash. Bind later callbacks to that same winner; never
silently switch execution settings while resuming.

Only then may the unchanged serial continuation machinery submit toward 100
**total** epochs, preserving parent AdamW/RNG lineage and the separate fresh-prior
RNG. Continue to recheck checkpoint/source/data consistency, completed segment
health and official user quota; stop on failure. Preserve the epoch 35 parent
and final held-out evaluation obligations. The earlier official quota margin
was small, so concurrent source114 output growth must be considered by actual
fresh checks, not assumed reserved space. No extra cleanup is implied here.

The documentation owner performs no model tests or HPC calls. Packaging,
allocated checks, actual submissions/cancellations and all successful/failed
outcomes will be entered in the ledger when evidence is available.

Historical source116 roots are `/work3/s234818/latentmol/releases/source-116-chemistry-speed-selection-attempt001`,
`/work3/s234818/latentmol/submissions/source116-chemistry-speed-selection-attempt001`
and `/work3/s234818/latentmol/runs/source116-chemistry-speed-selection-attempt001`.
The external evidence collector was extended before first export to retain
candidate `.out`/`.err` files. Original preparation/archive/operators remain
preserved; `collect-source116-first-suite-v2.py`, SHA256
`9e69ba78c547a584b8ccb43d11e591609c1dc8e9f2b8580389d6651aae7ac6cb`,
is separately bound by
[collector-recovery.json](../outputs/hpc-results/source116-first-suite-001/submissions/source116-chemistry-speed-selection-attempt001/collector-recovery.json).
A future preparation existence-check improvement was made after successful
packaging; packaging was not rerun and no sealed source changed. These were
export/preparation-operator refinements, not failed numerical experiments.
