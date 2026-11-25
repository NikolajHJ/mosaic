# Atom-preserving chemistry continuation of the triangle VAE

The [sulfur mechanism audit](84_sulfur_bias_in_validity_guidance.md) explains why
this policy removes the old live element-cap shortcut while retaining possible
shared-decoder, target-refresh and composition effects. It does not change this
protocol or establish a new experiment outcome.

**Latest status, 17 October 2026 22:57 DTU-local:** the old pending source118
benchmark/callback were intentionally retired. Replacement source121 attempt002
CPU29747653 passed715 checks/four CUDA-only skipped and its CPU repair profile;
16-slot/12-worker GPUbenchmark29747654 and callback29747655 remain PEND. The
five-candidate suite retains native/loss/retention gates; future segments are
24h allocated/23h internal. No native timing, selected ratio or training exists.
Last9.85GiB free is below the unchanged10.125GiB initial training floor, which
must pass a fresh check before training. [Verified evidence](../outputs/hpc-results/source121a002-first-suite-002/local-verification.json),
[current policy](61_chemistry_execution_selection.md) and [ledger](05_experiment_ledger.md).

**Historical status, 17 October 2026 21:03 DTU-local:** GPU29745548 and callback29745549
remain PEND. The completed CPU/profile gate remains valid; there are no native
speed measurements, chosen ratio/winner or new training results. Current eligible
host capacity is full/reserved; no ETA is confirmed. No VAE jobs/settings changed.
[Verified capsule005](../outputs/hpc-results/source118-first-suite-005/local-verification.json)
and [ledger](05_experiment_ledger.md).

**Source118 CPU/profile passed, 17 October 2026 13:41 DTU-local:** CPU29745547
DONE, 669 passed/four CUDA-only skipped in70.48s and a persisted passing profiler/
CPU gate. GPU29745548/callback29745549 PEND; all four CUDA checks stay
mandatory in the native stage. This is a test-only recovery
of source117's directory-setup error, with production/configurations/assertions
unchanged. Its failed gate stays preserved and blocked GPU/callback are confirmed
retired. No native timing, speed/winner, adopted 1:2 protocol or
training start exists yet. [Verified capsule003](../outputs/hpc-results/source118-first-suite-003/local-verification.json),
[docs61](61_chemistry_execution_selection.md) and [ledger](05_experiment_ledger.md).

**Historical source117 CPU failure, 17 October 2026 13:33 DTU-local:** one fixture
failed /668 passed/four skipped before profiler/native execution. A `__new__`
fixture omitted the directory created by the production constructor; its
positive provenance case failed writing the selection receipt. No native
speed/winner/training result exists. Pending GPU/callback retirement was
requested, and a test-only source118 recovery is being prepared without
changing science/configuration/assertions or approving the failed gate.
[Verified capsule002](../outputs/hpc-results/source117-first-suite-002/local-verification.json)
and [ledger](05_experiment_ledger.md).

**Historical source117 submission, 17 October 2026 13:32 DTU-local:** CPU29745524 RUN,
GPUbenchmark29745525/callback29745526 PEND. CPU preflight/profile is right-sized
to16 slots/64GiB; all13 scientific configurations, gate/selection policy and
the32-slot GPU suite remain unchanged. Source116's three unstarted pending jobs
were intentionally retired and confirmed EXIT, preserving their history.
No CPU pass, native result, speed/winner, adopted1:2 protocol or training start
is established. Source114 is untouched.
[Verified capsule001](../outputs/hpc-results/source117-first-suite-001/local-verification.json),
[docs61](61_chemistry_execution_selection.md) and [ledger](05_experiment_ledger.md).

**Historical source116 submission, 17 October 2026 13:27 DTU-local:** actual
CPU 29745514/GPU benchmark 29745515/promotion 29745516 are all PEND in verified
capsule001; no new CPU/native pass, winner or training start exists yet. Thirteen
candidate plans are bound to the same epoch 35 parent/source/data; the serial
suite prioritizes 1:3 below ten hours then conditionally evaluates 1:2. Every
scientific/health/runtime gate and quota check remains required. The 1:2 protocol
has not been adopted. Resources, exact mixture semantics, source hashes and
receipts are in [docs61](61_chemistry_execution_selection.md),
[verification](../outputs/hpc-results/source116-first-suite-001/local-verification.json)
and [ledger](05_experiment_ledger.md). Source112's failed 22h benchmark and
source115's passed CPU/intentional pending-worker retirement remain preserved.

**Earlier speed-selection preparation, 17 October 2026:** the user approved
overlap/worker/microbatch benchmarks with 1:3 preferred below ten hours/epoch,
otherwise a declared 1:2 fallback, always retaining all native gates. Pending
source115 benchmark 29745445/callback 29745446 were intentionally retired and
observed EXIT at 13:15; its CPU/profile pass remains preserved. No source116
submission or training start is established yet. Exact fallback loss/population
weighting and deterministic selection are in
[docs61](61_chemistry_execution_selection.md); all historical source112/115
outcomes and the epoch 35 parent remain preserved.

**Runtime-only successor submitted, 17 October 2026:** source115 keeps the
same loss, 1:3 mixture, all 18 gates, model/data/split and epoch 35/update 230860
parent/optimizer/RNG lineage. Its only protocol delta is 71h internal segment
time (`255600s`) with 72h scheduler allocation and 1h shutdown reserve; the same
1.5× plus 1800s fit formula is retained. Seven runtime/config/test files change;
448 other inherited scientific files remain identical. This does not relabel
source112's failed 22h readiness as a pass.

Actual latest jobs, 17 October 13:02 DTU-local: **CPU 29745444 DONE / gate passed**,
**native benchmark 29745445 PEND**, **promotion 29745446 PEND**. Verified capsule003
records 384 passed/four CUDA-only skips in 46.68s and profiler returncode 0 with a
persisted passing CPU gate. Native readiness and a training start remain pending.
Only after all fresh checks/quota pass may serial
continuation toward 100 total epochs start. The 1:2 fallback was not used;
original parent/checkpoints and source112 failure remain preserved. Official
free quota was about 10.48 GiB versus the strict 10.125 GiB initial guard and
must be freshly rechecked.
[Preparation](../outputs/hpc-submissions/source115-preparation.json),
[verified capsule003](../outputs/hpc-results/source115-first-suite-003/local-verification.json)
and [ledger](05_experiment_ledger.md) retain exact hashes and commands.

Speed options discussed afterward are saved-result analysis, not a protocol
change: source115 remains 1:3. See
[docs58](58_atom_preserving_repair_performance.md#saved-result-speed-analysis-17-october-2026)
for timing definitions and explicit loss-mixture caveats before any 1:2 fallback.

**Earlier VAE status, 17 October 2026 12:44 DTU-local:** **no current 1:3 chemistry
training is queued or running**. The live job list contains source11429741066
PEND and historical 28 September job29510465 PEND; that historical job is not
this continuation and was left untouched. The verified source112 outcome below
remains 17/18 gates passed, blocked only by epoch-runtime fit (22.4h projected
versus 22h segment; about 34.1h required reserve). The epoch-35 parent remains
preserved. No runtime fix or new job was submitted in this read-only status turn.
[Live receipt](../outputs/hpc-results/vae-status-20261017-124444.json) and
[ledger](05_experiment_ledger.md).

**Verified GPU outcome retrieved, 17 October 2026:** source112 benchmark
29736546 passed four CUDA tests (12.78s), authenticated the epoch-35 parent and
passed exact native resume and reference/optimized execution parity, including
model, gradients, optimizer and RNG. **17/18 benchmark gates passed; only
`complete_epoch_fits` failed.** Two updates took 12.2312/12.0329s, with
8.1965/7.9982s search, projecting an 80,621.44s epoch (22.4h). The unchanged
`1.5*epoch_seconds+1800 < 79200` fit rule needs about 34.1h reserved, exceeding
the 22h segment. Short-benchmark quality/retention/composition/sulfur checks
passed within their measured scope, not as proof of generation improvement.
The benchmark discarded its weights; callback29736547 persisted `stopped` and
submitted no training segment. **The 1:3 continuation has not started; the
epoch-35 parent remains preserved.** No settings, allocations, gates or jobs
were changed in this retrieval turn.
[Verified capsule003](../outputs/hpc-results/source112-first-suite-003.tar.gz),
[status summary](../outputs/hpc-results/source112-first-suite-003/status-summary.json)
and [ledger](05_experiment_ledger.md) retain exact identities and commands.

**Earlier access-limited observation, 17 October 2026:** source112 benchmark 29736546 and callback
29736547 were observed EXIT in a later 16 October live query, but logs/cause have
not been downloaded/reviewed. Today's SSH accepts the correct loaded key then
requires additional password authentication, so no fresh remote result is
available. Earlier pending/no-training statements below describe those dated
snapshots; they do not establish today's job or training state. The cause and
current chemistry-continuation outcome remain unknown. See the
[ledger](05_experiment_ledger.md).

**Historical CPU/profile snapshot, 16 October 2026:** source112 CPU 29736545
completed 333 passing checks / four CUDA-only skips and passed the identity-bound
native-parent CPU teacher profiler. Identical ordered outputs/checks/targets
were 26.35× faster with twelve workers on the small diagnostic cohort. This is
not a full GPU training speedup or a native runtime pass. GPU 29736546 and
callback 29736547 were pending; **no 1:3 training start was established at that snapshot**. Source/parent/loss/
ratio remain unchanged. Completed lossless archival preserved 73 historical
files; the official quota now has 10.78 GiB available, exceeding the unchanged
10.125GiB initial reserve at that reading. Native readiness and a fresh quota
check are still required before promotion. [Verified capsule002](../outputs/hpc-results/source112-first-suite-002.tar.gz),
[archival evidence](50_hpc_space_cleanup.md)
and the [ledger](05_experiment_ledger.md) preserve exact proof and limitations.

**Historical source112 submission snapshot, 16 October 2026:** actual **CPU 29736545**,
**GPU benchmark 29736546**, **promotion 29736547** are submitted. All results
are pending; **no 1:3 training has started**. Three fixture/helper corrections
leave the loss, ratio, search policy, parent and data unchanged. Source111's
blocked GPU/callback are confirmed owner-canceled; its CPU failure remains
history. Exact reference parity, practical runtime, retention and quota must
still pass before continuation toward 100 total. [Preparation](../outputs/hpc-submissions/source112-preparation.json)
and the [ledger](05_experiment_ledger.md) retain the new immutable identities.

**Earlier preflight failure, 16 October 2026 13:17 DTU-local:** source111
CPU 29736537 exited with nine failed / 323 passed / four CUDA-only skipped.
Controller/RDKit fixture assumptions and CPU snapshot aliasing in the new
disposable parity helper are being corrected for a new immutable attempt.
The profiler/native GPU gate did not run; root requested cancellation of the
blocked GPU 29736538 and callback 29736539. **No 1:3 training started.** Loss,
ratio, search policy and epoch 35 parent remain unchanged; no failure is waived.
[Verified capsule001](../outputs/hpc-results/source111-first-suite-001.tar.gz),
[docs58](58_atom_preserving_repair_performance.md) and the
[ledger](05_experiment_ledger.md) retain the complete failure scope.

**Acceleration suite submitted, 16 October 2026:** new source111 has actual
**CPU 29736537**, **native GPU benchmark 29736538** and **promotion 29736539**
submissions. This is pending validation, **not a 1:3 training start**. Exact
reference teacher/full-update parity and all existing native/runtime/retention/
quota gates must pass before automatic serial continuation from epoch 35 to100
total. Scientific loss,1:3 ratio, data/configuration and parent identity remain
unchanged. [Preparation](../outputs/hpc-submissions/source111-preparation.json),
[docs58](58_atom_preserving_repair_performance.md) and the
[ledger](05_experiment_ledger.md) retain exact new identities and source110's
failed runtime benchmark.

**GPU resume checks passed; runtime gate blocked training, 16 October 2026
12:56 DTU-local:** source110 benchmark **29735032 EXIT**, after **all four CUDA
fixtures passed in 16.16 seconds**, original parent authentication passed and
the actual VAE's exact model/AdamW/RNG next-update replay passed. The device
comparison bug is fixed on native hardware. **16/17 gates passed**; only
`complete_epoch_fits` failed. Two full updates took **236.534 / 232.027 seconds**,
with **232.514 / 228.017 seconds (about 98%)** in repair search, projecting
**about 17.9 days per epoch**. Short-check gradient, repair coverage, memory,
reconstruction, composition and sulfur gates passed; this does not establish
long-run chemistry benefit.

The benchmark discarded its weights. Callback **29735033 EXIT/stopped**;
**no 1:3 training has started**, and the preserved VAE remains epoch 35.
Repair-search runtime must be addressed before the unchanged 100-total-epoch
continuation is practical; no gate was waived or new job submitted during
this result retrieval. [Verified capsule004](../outputs/hpc-results/source110-first-suite-004.tar.gz),
[native gate](../outputs/hpc-results/source110-first-suite-004/runs/source110-chemistry-resume-attempt001/benchmark_gate.json)
and the [ledger](05_experiment_ledger.md) retain exact outcomes and identities.

**Queue reconfirmed, 16 October 2026 10:44 DTU-local:** **GPU benchmark
29735032 PEND** for available physical GPU capacity; **promotion 29735033
PEND** on its expected dependency. GPU checks have not run, so **the 1:3
continuation has not started**. The preserved CPU result remains 192 passed /
four CUDA-only skipped. [Verified capsule003](../outputs/hpc-results/source110-first-suite-003.tar.gz)
and the [new queue record](../outputs/hpc-results/source110-queue-status-002.json)
confirm the unchanged source110 identities and waiting state. The authenticated
epoch 35 parent and automatic continuation toward 100 total epochs remain
conditional on actual GPU/native checks, quota and serial health; no new job
or code change was made in this status retrieval.

**Recovery CPU checks passed, 16 October 2026 10:00 DTU-local:** source110
**CPU 29735031 DONE**, **192 passed / four CUDA-only skipped in 15.36 seconds**
on the allocated Python 3.11.9 worker. **Native benchmark 29735032 PEND** and
**promotion 29735033 PEND**; no 1:3 training start or native replay pass yet.
Those four CUDA fixtures must run, without skips, in the GPU allocation before
the actual VAE replay, repair-coverage/runtime/composition/retention gates.
Epoch 35 remains the authenticated parent, with 65 remaining epochs to 100 total
conditional on native checks/quota and serial health. [Verified capsule002](../outputs/hpc-results/source110-first-suite-002.tar.gz),
[CPU gate](../outputs/hpc-results/source110-first-suite-002/runs/source110-chemistry-resume-attempt001/cpu_gate.json)
and the [ledger](05_experiment_ledger.md) retain exact proof and identities.
The [latest queue record](../outputs/hpc-results/source110-queue-status-001.json)
shows the GPU benchmark waiting for available physical GPU capacity on one
qualifying host; the callback is waiting on that benchmark as intended.

**Device-comparison recovery submitted, 16 October 2026:** new immutable
source110 has actual **CPU 29735031**, **native GPU benchmark 29735032** and
**promotion callback 29735033** submissions. Results are pending; **no 1:3
training start is established**. Exact tensor comparison uses detached CPU
copies with shape/dtype/layout/value equality, without moving live AdamW state
or relaxing native next-update/RNG parity. Four mandatory CUDA replay/failure
fixtures precede the real VAE benchmark in the same GPU allocation. The callback
allows only bounded scheduler `RUN` propagation and still requires **DONE**.

The authenticated epoch 35 /230,860-update parent, data/configuration, 1:3
objective and **100 total epoch** endpoint are unchanged; 65 remaining epochs
may start and continue serially only after all native/quota/health gates pass.
All 135 original source098 Python files and source107 failure evidence remain
preserved. [Source110 preparation](../outputs/hpc-submissions/source110-preparation.json),
[sealed plan](../outputs/hpc-submissions/source110-chemistry-plan.json) and the
[ledger](05_experiment_ledger.md) retain exact identities and pending scope.
This receipt-stage update does not claim a passed CPU or native benchmark.

**Native benchmark failed; no chemistry training, 16 October 2026:** source107
**29728631** passed original epoch 35/config/data/runtime/AdamW/RNG parent
authentication with **no differences**, so the strict JSON fix cleared the old
setup gate. It then failed after **55.578 seconds** in the disposable resume
comparison: `nested_equal` called `torch.equal` on CUDA/CPU tensors after reload
(`chemistry_continuation.py:640`, called from674). This is a device-handling
error in the comparator, not evidence of numerical resume mismatch, sulfur
drift or model instability. Probe updates were disposable and were not promoted
to a training segment. Coverage/runtime/retention and exact next-update parity
remain unpassed; **no 1:3 continuation has started**.

The callback **29728632 stopped** and submitted no training; its decision saw
predecessor `RUN`, while persisted benchmark evidence independently establishes
failure. The preserved real-only parent remains epoch 35/230,860 updates.
[Benchmark gate](../outputs/hpc-results/source109a002-pointwise-round-003/runs/source107-json-config-chemistry-attempt001/benchmark_gate.json),
[traceback](../outputs/hpc-results/source109a002-pointwise-round-003/runs/source107-json-config-chemistry-attempt001/benchmark/20261015T142616Z_chemistry-benchmark_594d9142/traceback.txt)
and [verified capsule003](../outputs/hpc-results/source109a002-pointwise-round-003.tar.gz)
retain actual results; the [ledger](05_experiment_ledger.md) preserves identities
and prior failures. No fix, rerun or new submission was made in this status turn.

**Queue reconfirmed, 15 October 2026 11:17 DTU-local:** source107 **CPU
29728630 DONE / 169 passed**; **native benchmark 29728631 PEND** and
**promotion 29728632 PEND** remain queued. **No native parent/next-update/RNG
replay pass or 1:3 training segment exists yet**. The strict JSON fix has
allocated regression evidence; native coverage/runtime/composition/retention/
quota gates still must pass before the 65 remaining chemistry epochs may start.
[Verified recovery-round capsule](../outputs/hpc-results/source108-recovery-round-001.tar.gz),
SHA256 `aaa89cab85640d51742995f09ec93b1efb5c14a8d37d56e985539633dcfb0c73`,
retains its unchanged source107 plan/CPU gate and fresh scheduler state alongside
the separate gradient suites. [Ledger](05_experiment_ledger.md) records exact
scope; no parent/scientific settings or failed outcomes were changed.

**Corrected source107 CPU gate passed, 15 October 2026 10:49 DTU-local:**
new immutable strict-JSON recovery is uploaded/authenticated. **CPU 29728630
DONE**, **169 passed in 14.59 seconds**, return code 0; **native benchmark
29728631 PEND**, **promotion 29728632 PEND**. No native checkpoint/optimizer/
RNG replay pass, native repair coverage/runtime/retention result or full 1:3
training is established yet. The callback retains every native/quota gate
before submitting training, then serial health checks toward 100 total epochs.

Archive/manifest/plan SHA256 are
`488a20adc3e84a7a4f5d0ea94cfcf5c1202f31900c3c978aec3d0c31980f8848` /
`6a64e8df4b174c2fc30013e09e0f0b048b2c3df7bedba4c26f603bc418741fec` /
`7927a512f89087065fe05c59633637f586fd806686f26681974ab589398a0083`.
Only chemistry implementation/its fixture changed; all 135 original source098
Python files and the parent/data/configuration/scientific protocol are preserved.
The new plan changes only archive, manifest, release and campaign identities;
epoch-35 parent still leaves 65 epochs to 100 total. [Capsule001](../outputs/hpc-results/source107-first-suite-001.tar.gz),
SHA256 `c54518fabcaa697daff371b4cae8138ad45d17aa2cc3138a32f519c6f6ccdec4`,
has 46 safe members / 45 verified files at
`2026-10-15T08:49:28.820974+00:00`. [CPU gate](../outputs/hpc-results/source107-first-suite-001/runs/source107-json-config-chemistry-attempt001/cpu_gate.json)
and the [ledger](05_experiment_ledger.md) retain actual outcomes;
[docs55](55_trajectory_and_parent_identity_recovery.md) explains the diagnosed
JSON/YAML error and strict correction. Earlier failures remain unchanged.

**Specific parent-identity cause identified, 15 October 2026:** allocated
CPU metadata **29728564 DONE** found one difference: canonical JSON numeric
`weight_decay: 1e-05` became the string `"1e-05"` when the `.json` parent
configuration was read with the historical YAML loader. The checkpoint's
epoch 35 / 230,860 updates, all other compared config/data/stored-runtime
fields and AdamW/RNG lineage remain intact. New chemistry code parses this
artifact strictly as JSON, preserves exact types and gives named mismatch
diagnostics; **new allocated regressions/native readiness are still pending**.
This does not start training, weaken equality checks or establish the failed
GPU allocation's unlogged runtime. [Docs55](55_trajectory_and_parent_identity_recovery.md)
and the [ledger](05_experiment_ledger.md) record verified cause, the failed
initial metadata command and the remaining gates. Original parent and all
previous failure evidence stay preserved.

**Native readiness failed before chemistry updates, 14 October 2026:** GPU
benchmark **29712276 EXIT 1**, 53-second allocation. Setup rejected the parent
with `ValueError: Parent completed epoch/config/data/runtime differs` at
`chemistry_continuation.py:196`; the native update, repair coverage, runtime
and retention measurements were not reached. This is a parent-identity gate
failure, not observed model instability, sulfur drift or a failed chemistry
training result. The generic error does not identify which compared field
differs; that needs a separate diagnostic.

Promotion **29712277 EXIT 1**, one second, recorded `status: stopped` because
the predecessor benchmark ended EXIT. It correctly submitted no full training.
The original epoch-35 parent and all failed evidence remain preserved. No new
chemistry job was submitted or gate loosened during this result retrieval.
[Verified capsule003](../outputs/hpc-results/source104a005-first-suite-003.tar.gz),
SHA256 `5bcb8b2539486127b2a9603ab421d080aa8c23fcdc37b429b384c39e07fd04db`,
has 63 safe members / 62 verified files at
`2026-10-14T13:26:21.582678+00:00`; [benchmark failure](../outputs/hpc-results/source104a005-first-suite-003/runs/source104-atom-preserving-chemistry-attempt005/benchmark_gate.json)
and [stopped promotion](../outputs/hpc-results/source104a005-first-suite-003/runs/source104-atom-preserving-chemistry-attempt005/advance-after-benchmark.json)
retain actual outcomes. The [ledger](05_experiment_ledger.md) records paths,
source/plan/parent identity, exact logs and numerical scope. **The 1:3 phase
has not started**, and 65 epochs to 100 total remain conditional on corrected,
authenticated readiness rather than reuse of the CPU pass alone.

**Current readiness, 14 October 2026 13:20 DTU-local:** the new epoch-35
plan005 retains its allocated **112-pass CPU gate**. **Native benchmark
29712276 PEND** and **promotion callback 29712277 PEND**; no native repair
coverage/runtime/retention result or 1:3 chemistry training exists yet.
The root's live detailed queue read confirms the CPU dependency is fulfilled
and reports `Job slot limit reached: 3 hosts`; the callback correctly awaits
`ended(29712276)`. This is a queue delay, not an observed chemistry runtime
failure, and does not supply a guaranteed start time.

The [fresh verified capsule002](../outputs/hpc-results/source104a005-first-suite-002.tar.gz),
SHA256 `2eb7261964db135c1be19f3b161f57ba51cc4dd90c3dc7ef15327e8230f6441c`,
has 51 safe members / 50 verified files, exported
`2026-10-14T11:20:51.055553+00:00`. It preserves the epoch-35 parent,
new immutable plan, original intentional stop, actual CPU pass and pending
native/callback snapshot. Quota reports 88.16 GiB used of 100 GiB; the
promotion must still authenticate available quota and all native gates.
The intended 65 remaining epochs to 100 total remain conditional. No source,
parent, loss, data or configuration was changed or new job submitted during
this read-only status retrieval. [Ledger](05_experiment_ledger.md) records
the separate completed contact diagnostic; it does not waive this phase's
readiness gates or the Boltz/DeFoG optimization guards.

**Current readiness, 13 October 2026 23:00 DTU-local:** new epoch-35 plan005
**CPU 29712275 DONE**, **112 passed in 13.02 seconds**, return code 0,
with exact parent/source/plan identity. **Native benchmark 29712276 PEND**
and **promotion callback 29712277 PEND**. No native benchmark or full 1:3
training has run; **65 remaining epochs to 100 total** are conditional on
all native/quota/serial health checks. The original real-only run/controller
are intentionally stopped, preserving published epoch 35 / 230,860 updates
and discarding partial epoch 36.

[Verified capsule001](../outputs/hpc-results/source104a005-first-suite-001.tar.gz),
SHA256 `10727f4375a5ebd96328b9a619eda01feb1ccbc9e6d723a70aa6c650e7c9c543`,
has 51 members / 50 verified files at 21:00:06.268379 UTC, including the
stop marker, new snapshot/plan/receipts and allocated CPU outcome.
[CPU gate](../outputs/hpc-results/source104a005-first-suite-001/runs/source104-atom-preserving-chemistry-attempt005/cpu_gate.json)
and [exact log](../outputs/hpc-results/source104a005-first-suite-001/runs/source104-atom-preserving-chemistry-attempt005/allocations/20261013T205742Z_source104-chemistry-cpu_1395677f/command.log)
establish engineering readiness only. The [ledger](05_experiment_ledger.md)
retains commands, environment and all previous failures/cancellations.
Local verification used Python 3.12.14 for bytes/metadata only; numerical
fixtures ran on allocated HPC. No scientific source/loss/data/config changed.

**Epoch-35 readiness suite submitted, 13 October 2026:** new attempt005
has **CPU 29712275**, **native GPU benchmark 29712276** (`done(29712275)`)
and **promotion callback 29712277** (`ended(29712276)`). The source098
real-only run/controller and the old epoch-31 benchmark/promotion are stopped.
Exact scientific attempt004 source/protocol/data/configuration is reused;
the new plan binds the immutable epoch-35 parent and unique campaign005.
These jobs are submitted, not passed. **No chemistry training has started**;
the remaining 65 epochs to epoch 100 total are conditional on the new CPU/
native/quota gates and subsequent serial health checks. The old 112-pass
CPU result remains historical; this new plan gets its own gate. The
[ledger](05_experiment_ledger.md) retains the new submitted IDs and recovered
pre-upload metadata submitter correction.

**Current parent and stop, 13 October 2026 22:54 DTU-local:** real-only
source098 GPU 29703164 was intentionally interrupted, callback 29703165
canceled, and the old epoch-31 chemistry benchmark/promotion 29712043/29712044
canceled before execution. All four have EXIT receipts. The
[new stop/snapshot record](../outputs/hpc-submissions/source104a005-parent-snapshot.json)
preserves all source098/original artifacts and immutable **epoch 35 /
230,860 updates**, SHA256
`19f438222e454ef53169f6d81082335ee7e24902b1b7cf5e0f18eced4b8e92ed`.
Partial epoch 36 is discarded. This is user-directed interruption, not a
failed health check, model instability or an epoch-100 real-only endpoint.
The original raw run JSON is retained despite its stale `running` field.
New **attempt005** uses the unchanged scientific attempt004 release,
protocol and dataset with a new parent-bound plan, targeting **65 remaining
epochs to 100 total** if all native gates pass. It is prepared, not yet
submitted or numerically validated. Historical epoch-31 plans/results remain
below; they are not silently modified.

The new [plan](../outputs/hpc-submissions/source104a005-chemistry-plan.json)
SHA256 is `b08ec037c3d75136c1ce73fb58a4ec0c2e15377a6bf1af6c41e1de8e8fc58792`.
Parent path is
`/work3/s234818/latentmol/submissions/source104-chemistry-continuation-attempt005/parent.pt`,
copied from immutable `epoch_0035.pt` in the original segment 002 run.
New campaign/control are
`/work3/s234818/latentmol/runs/source104-atom-preserving-chemistry-attempt005`
and `/work3/s234818/latentmol/submissions/source104-chemistry-continuation-attempt005`.
The old real-only-to-100/final-evaluation obligation is superseded; final
chemistry test access remains gated by its authenticated epoch-100 endpoint.
The [ledger](05_experiment_ledger.md) retains stop order, exact scheduler
receipts, parent/config/source/data hashes and the recovered local
Windows-path metadata-packager failure.

**Latest user direction, 13 October 2026:** stop the original real-only
source098 training and automatic continuation, then repin the 1:3 chemistry
phase to its **most recent completed authenticated checkpoint**, targeting
**100 total epochs**. The earlier requirement to retain a running real-only
baseline through epoch 100 and final evaluation is superseded. Preserve all
source098/original checkpoint/evidence files; the stop is intentional, not
model instability or a completed epoch-100 comparison. The old queued
epoch-31 benchmark/promotion will be canceled before the baseline stop.
Actual receipts, stopped epoch/checkpoint and a new immutable chemistry
plan remain pending; the last earlier verified baseline epoch was 35.
All native chemistry readiness/coverage/runtime/retention/quota gates and
serial automatic continuation remain required. **The historical epoch-31
identities/settings below are retained, not silently rewritten.**

**Latest status, 13 October 2026 22:43 DTU-local:** attempt004's allocated
CPU gate still has **112 passes**; **native benchmark 29712043 PEND** and
**promotion callback 29712044 PEND** remain unexecuted. The live queue read
reports 9,488 seconds eligible pending under `Job slot limit reached: 3 hosts`,
with `done(29712042)` fulfilled. The
[fresh verified capsule002](../outputs/hpc-results/source104a004-first-suite-002.tar.gz),
SHA256 `2056373d8e89f8815c94d512524be806c5596c865cc60f96cb5f51885d4950df`,
has 29 members / 28 verified files at 20:42:59.085445 UTC. Old CPU scheduler
history has aged out; durable completion remains retained. No native repair
coverage/runtime, chemistry training or generation benefit is established.
The unchanged source098 baseline has now reached **35/100 epochs**, while
this new branch still uses the authenticated **fixed copied epoch-31 parent**.
No source, parent, loss, data, submission or scheduler state changed during
the status-only retrieval. The [ledger](05_experiment_ledger.md) retains
all three fresh capsules and the separate contact diagnostic still queued.

**Current readiness, 13 October 2026 20:05 DTU-local:** immutable source104
attempt004 **CPU 29712042 DONE**, **112 checks passed in 12.59 seconds**,
return code 0 and matching source/parent/plan identity. **Native GPU benchmark
29712043 PEND**, with its CPU dependency fulfilled; **promotion callback
29712044 PEND** awaits the benchmark's end. Full chemistry training has
**not** been submitted, and no native coverage/runtime or generation
improvement is established yet. The unchanged source098 baseline remains
running independently.

At the root's separate **20:08 DTU-local** queue check, 29712043 is eligible
PEND under `Job slot limit reached: 3 hosts`; the CPU dependency is fulfilled
and callback 29712044 is correctly waiting on `ended(29712043)`. No access
issue or failed native allocation is observed. There is no guaranteed start
time. If the native benchmark and authenticated quota checks pass, the
callback automatically starts the remaining **69 chemistry epochs toward
epoch 100 total**, then serially continues only while health checks pass.

[Verified capsule001](../outputs/hpc-results/source104a004-first-suite-001.tar.gz),
SHA256 `686055f0b7383aa3954d5b8336b64a44ea7cdbfb30835efb3e2a08309afc43d8`,
contains 29 safe members / 28 verified files at 18:05:52.010928 UTC.
The [CPU gate](../outputs/hpc-results/source104a004-first-suite-001/runs/source104-atom-preserving-chemistry-attempt004/cpu_gate.json)
and [exact command log](../outputs/hpc-results/source104a004-first-suite-001/runs/source104-atom-preserving-chemistry-attempt004/allocations/20261013T180500Z_source104-chemistry-cpu_93c3d158/command.log)
retain allocated engineering success. Original attempt003's CPU failure
remains preserved; its blocked benchmark/callback are now confirmed EXIT
without execution. [Ledger](05_experiment_ledger.md) retains commands,
environment, requests, failures, cancellations and pending native scope.

**Fixture-only recovery submitted, 13 October 2026:** new source104 attempt004
has **CPU 29712042**, **GPU benchmark 29712043** (`done(29712042)`) and
**promotion callback 29712044** (`ended(29712043)`). Checks have not yet
been verified; no full chemistry training has been submitted. Only the
malformed-target fixture changed; scientific repair/training/protocol and
parent/data remain unchanged. Original attempt003 failure remains below;
termination requests for its blocked GPU/callback were accepted, with final
scheduler capture pending. The [ledger](05_experiment_ledger.md) retains
attempt004 bindings and actual submitted IDs.

**First allocated CPU gate failed, 13 October 2026:** source104 attempt003
submitted CPU checks **29712033**, native GPU benchmark **29712034** with
`done(29712033)`, and promotion callback **29712035** with
`ended(29712034)`. The CPU job **EXIT** after **111 passed / one failed in
14.51 seconds**. The failed fixture attempts to resize intentionally
non-resizable target storage before checking shape rejection. The full
gate is failed; native GPU benchmark and callback have not run, and the root
is canceling those blocked jobs. A new immutable fixture-only attempt004
is being prepared. **Full chemistry training has not been queued.**
No actual pretrained-GPU coverage/runtime or improved generation is established.
The [verified capsule001](../outputs/hpc-results/source104a003-first-suite-001.tar.gz),
SHA256 `06987ee3d6ab281c8c4e38cf7b526da2df4edd577f363fa7f4dceb0670131f4b`,
has 30 members / 29 verified files at 18:01:00.752588 UTC; the
[ledger](05_experiment_ledger.md) retains the exact command, environment,
failed trace, source/plan identity and corrective scope.
The source098 baseline continues under its unchanged reconstruction-plus-KL
objective, with its current segment 002 **29703164**; historical first
segment **29694051** is already completed.

**Authorization and preparation, 13 October 2026:** the user authorized
implementation of an atom-preserving repair loss and a composition regularizer,
allocated-HPC benchmarking, and a **1:3 real-to-fresh-random continuation**
from the current triangle VAE if its checks pass. The user selected
**100 total epochs**, counting epochs already completed by the parent VAE;
this is not permission for another 100 epochs. Source098, the original
epoch-89 VAE, and all historical decoder-only studies are preserved.
This document records the new branch; it does not change the bindings or
loss of the reconstruction-plus-KL source098 campaign.

The following initial entry is a historical preparation record, not a passed benchmark,
or efficacy result. The current completed **epoch-31 parent** has been copied
as a byte snapshot, with **204,476 optimizer updates** and **69 epochs
remaining** to the absolute epoch-100 endpoint. The new sealed source/plan
and submitted first-suite job IDs are recorded above and below; measured
runtime and numerical checks remain pending.
The [snapshot metadata](../outputs/hpc-submissions/source104-parent-snapshot.json)
records the copied parent bytes as **not numerically validated**.
All numerical work remains on allocated DTU HPC. Local work is editing, static review,
immutable packaging, transfers and byte/metadata inspection only.

The new phase **trains both encoder and decoder**. It intentionally carries
the authenticated parent's AdamW moments and persisted RNG lineage across
the declared objective change, and creates a separate seeded fresh-prior RNG
whose state must be checkpointed and restored. This is a changed-objective
continuation, not an exact unchanged-source098 resume. No old
maximum-allowed-valence penalty is added to this phase.

The copied remote parent is
`/work3/s234818/latentmol/submissions/source104-chemistry-continuation-attempt001/parent.pt`,
SHA256 `34fa3dec01ceb140222ad865df938b273c8cde2a2835bad1f6bee2da43e57659`.
It comes from source098's
`segments/002/20261013T054947Z_vae-train_9e484e3f` run. The copied
`parent-config.json` SHA256 is
`b6704c80c4f3308b5f20623eb0ef21fc39c4b7b1d0cd60d519a9990670b1b254`.
Byte authentication is not proof that the native new-phase load/resume gates
have passed. The source098 baseline continues with its unchanged objective;
the chemistry branch starts from the fixed copied epoch 31 and never modifies
that parent file.

## Why this is a new branch

The parent triangle VAE's epoch-25 deep validation panel reconstructs encoded
molecules well but produces few valid unfiltered outputs from direct
standard-normal prior draws: **47/4,096 argmax graphs** and **73/16,384
categorical draws**. These results use raw attempts in the denominator and no
repair. They establish a generation limitation under that particular sampling
policy, not its cause or a guarantee that this continuation will solve it.
The [large-corpus pipeline](52_large_corpus_triangle_vae.md) and ledger retain
the authenticated source100 capsule005 and original evaluation procedure.

The [historical dense decoder continuation](45_dense_chemistry_100_epoch_continuation.md)
increased random-prior validity but substantially increased expected sulfur
content. Its maximum-allowed-valence proxy lets element substitutions reduce
penalty without repairing the underlying bonding. Sulfur's legitimate expanded
valence must not be falsified to remove that shortcut. That experiment is a
different architecture/data/training regime and remains historical evidence,
not a matched causal comparison for this triangle continuation.

The new objective therefore first holds sampled atom identities, formal
charges and occupied slots constant while finding bond repairs. A separate
weak population composition penalty discourages wholesale element/charge
drift. Neither component asserts biological usefulness, synthesizability,
binding affinity, or globally nearest chemical validity.

The unchanged canonical representation is a padded **50-slot heavy-atom
topology graph**, with a 256-dimensional latent space. Its element vocabulary
is carbon, nitrogen, oxygen, fluorine, phosphorus, sulfur, chlorine, bromine
and iodine, plus `NO_ATOM`; charges and bond classes remain those bound to
the parent schema. Stereochemistry and 3D conformers are not added by this
chemistry phase. Native sanitization/connectedness and a schema roundtrip
establish validity under that representation, not stereochemical fidelity,
synthetic accessibility or stability of an isolated compound. A charged
candidate may have no discovered bond-only repair under the bounded policy;
charge preservation is a scope limit, not a proof of impossibility.

## One training update

1. Draw a logical batch of **256 real molecules** from the unchanged supported
   training split (1,688,539 supported training identities).
   Encode it, draw posterior latents using the existing VAE policy, and decode
   the same shared graph schema. Reconstruction and VAE KL remain supervised
   on these real molecules.
2. Draw **three fresh independent `N(0,I)` latent points per real example**:
   768 fresh prior points for a full logical batch. Preserve the exact 1:3
   ratio on the final short batch as well. Process all 1,024 full-batch
   decoded candidates using microbatches of **128** to control VRAM.
   Each prior point is in the same 256-dimensional latent space. These are unconditional draws,
   not perturbed real encodings, selected successful points, rescaled prior
   vectors, or a fixed pool reused across the phase. Decode them using the
   same decoder. They have no fabricated reconstruction label or encoder KL.
   Any microbatch replay inside this optimizer update reuses these exact
   points and stochastic state; it does not draw a replacement random batch.
   Freshness means a new batch on the next update, with RNG state persisted
   so a resumed run does not repeat or skip its prescribed draws.
3. Outside autograd, take the canonical **argmax** candidate for each
   scheduled decoded graph. Symmetrize bond logits using the shared schema's
   rule before hardening. Store occupied slots, element labels, charges,
   unordered bond labels and exact probability/search identities. The teacher
   adds no categorical sampling RNG; fresh latent draws remain stochastic.
4. Run the declared bounded, probability-aware **bond-only** search against
   the same chemical-validity verifier. Each candidate retains exactly the
   original occupied slots, atom labels and charge labels. Count each
   undirected bond once, preserve symmetry and keep the diagonal empty.
   Search ordering uses detached decoder probabilities; chemical conversion,
   target search, acceptance and categorical decisions are outside the
   differentiable path.
5. Accept a fixed target only after schema, nonempty occupancy, connectedness,
   chemical validation and exact slot/atom/charge alignment checks pass.
   Already-valid candidates and unsuccessful searches contribute zero repair
   loss under the initial policy. A failed search is censored absence of a
   witness, not proof that the sampled atom/charge assignment admits no valid
   molecule.
6. Compute full categorical repair-target cross-entropy from the **live
   decoder atom, occupied-charge and unordered occupied-bond probabilities**
   to each accepted stopped target, plus the separately
   normalized population composition and size/occupancy safeguards.
   Backpropagate through the learnable encoder and decoder under the joint
   training scope. Real reconstruction/KL and real decoded chemistry can
   update the encoder; unconditional random-point branches bypass it and
   update the decoder. No gradient flows through RDKit, discrete candidate
   sampling or the search trajectory.
7. Record the update's complete point counts, target availability, chemistry
   terms, composition/size drift, finite-gradient health and search/runtime
   budget consumption. Persist model, optimizer and all required RNG states
   at declared boundaries.

The reviewed initial coefficients, search caps and health thresholds are
specified below; their new immutable protocol identity and actual numerical
outcomes remain pending. The joint parameter scope, user-directed example
ratio and 100-total-epoch maximum are fixed. Both already-valid candidates
and unavailable targets have zero repair contribution; no identity cross-entropy is added
for already-valid graphs in the initial policy.

## What the repair gradient means

For an accepted graph witness `H`, the whole categorical target is fixed for
that update. The initial objective is the normalized negative log probability
of its entire fixed categorical graph under the current decoder:

\[
\ell(P;H)=-\frac{
\sum_{i=1}^{N}\log a_{i,x_i^H}
+\sum_{i\in O_H}\log c_{i,q_i^H}
+\sum_{i<j,\;i,j\in O_H}\log b_{ij,e_{ij}^H}
}{N+K+K(K-1)/2},
\qquad K=|O_H|,\quad N=50.
\]

All atom-slot labels contribute, including `NO_ATOM`. Charges contribute
only at occupied slots. Every unordered occupied pair contributes once,
including `NONE` bonds; self-pairs and padding bonds do not contribute.
The NLL uses live log probabilities rather than the probability floor used
for detached search ordering. Averaging these per-graph values includes
missing-target zeros. One valid witness does not establish a globally
closest valid graph. Probability-aware ordering can prefer edits already
supported by the decoder, but a finite beam/check/time budget can miss a
less costly witness or fail to find any witness.

The [teacher implementation](../src/latentmol/atom_preserving_chemistry.py)
restricts the existing probability-aware edit policy to bond labels. A
changed label's observed cost is
`1 + probability_weight * max(0, log(p_old/p_new))`, using the declared
floor only in this detached cost. Search prioritizes implicated bonds,
fragment links and probability regret and keeps bounded frontier states.
The receipt stores raw/witness label identities, source probability hash,
policy, candidate/update IDs, edit count, cost, checks and failure status.
Any cache reuse requires the exact same graph labels, probability identity
and restricted policy; new point/update IDs are still rebuilt. This is not
a fixed reused latent training pool or a target from a different decoder
probability state.

The initial bounded policy allows at most **three distinct bond-label edits**,
at most **32 checked candidates per depth**, and a **four-state beam**.
The detached edit-cost probability weight is 1.0 and its floor is `1e-12`.
The beam/check caps apply to each scheduled graph. They do not authorize
skipping graphs from the nominal 256:768 logical batch. Continuing past a
first valid witness is allowed within the same declared caps; the best
observed probability-weighted witness is retained. No exact minimum is
asserted after a capped layer or beam pruning.

Holding atom/charge labels fixed in search prevents the search from solving
overbonded carbon by replacing it with sulfur. **Bond-only edits do not mean
bond-only gradients**: for accepted witnesses, the unchanged atom, charge
and occupancy labels still supply identity cross-entropy, reinforcing the
candidate identity as its repaired bonds become more likely. Already-valid
candidates receive no self-target or identity NLL. Shared parameters can
change the resulting distributions as well. The population composition
guard discourages a global element/charge shift, but neither atom-preserving
targets nor this weak regularizer mathematically guarantees no sulfur drift.

Search outputs are labels for a surrogate training objective. They are not
substituted into raw generation evaluation or into the shared Boltz input,
and no Boltz/DeFoG model is trained by this experiment. Any resulting decoder
must be frozen before molecular latent optimization.

## Loss weighting and the composition prior

The initial protocol form is

\[
L=L_{\mathrm{reconstruction,real}}+
\beta L_{\mathrm{KL,real}}+
\lambda_e\left(\tfrac14 L_{\mathrm{repair,real}}+
\tfrac34 L_{\mathrm{repair,random}}\right)+
0.1\,JS_{\mathrm{element}}+0.05\,JS_{\mathrm{charge}}+
1.0\,L_{\mathrm{occupancy,prior}},
\qquad
\lambda_e=0.5\min\!\left(\frac{e-e_{\mathrm{parent}}}{3},1\right).
\]

For current attempt005, `e_parent=35`: absolute epochs 36, 37 and 38 use
repair coefficients one third, two thirds and all of the maximum 0.5.
Historical unexecuted epoch-31 plans instead selected epochs 32–34; their
bindings remain preserved. The real/random repair
weights remain 0.25/0.75 throughout. The occupancy safeguard is the mean
squared difference between current and frozen plan-bound parent occupancy
probabilities at each **same fresh prior point**; it has coefficient 1.0
and is not applied to an unrelated reference graph. It is averaged over
the full prior branch, not scaled threefold merely because that branch
contains three times as many points.

The copied source098 real-data reconstruction settings remain bond weight
2.0, charge weight 1.0, no-bond weight 0.1 and padding weight 0.2, with no
additional reconstruction-size term. KL uses the inherited maximum beta
0.001 and absolute-epoch 20-epoch warmup, already complete at the branch
start. The inherited learning-rate schedule uses 0.001 before epoch 61,
0.0003 at epoch 61 and 0.0001 at epoch 81; AdamW weight decay is `1e-5`.
Inherited trainer gradient clipping at norm 5 remains a numerical update
safeguard, not a repair validity claim or a solution to historical Boltz
finite-difference discrepancies. These inherited settings are preserved
deliberately and bound by the copied parent configuration.

The real and random repair means each include **all scheduled points**,
including already-valid candidates, unavailable/censored targets and zero
contributions. Do not normalize only by accepted witnesses: doing so changes
the objective with search coverage. A threefold increase in random batch
size alone does not impose a 0.25/0.75 objective mixture when branch means
are added with equal coefficients. The initial implementation uses the full
categorical graph normalization `N + K + K(K-1)/2`; it does not rebalance
connected versus absent-bond labels. Any later alteration must be explicit
in the sealed protocol.

The composition regularizer compares probability-weighted **population**
element and formal-charge distributions to empirical frequencies derived
from the authenticated supported training corpus. A Jensen–Shannon
divergence is a bounded divergence between distributions; it is not a
chemical-validity score. Its reference counts, smoothing, occupancy
conditioning, branch weights and coefficient must be saved. Count invalid
and unsuccessful outputs too: conditioning composition on valid outputs
would conceal drift in most generated attempts. Do not replace empirical
reference frequencies with invented sulfur limits.

Aggregate expected element counts exclude `NO_ATOM`; expected charge counts
are weighted by each slot's occupancy probability. Normalize the aggregate
count vectors once over the whole optimizer-update population, not once per
molecule or microbatch. The implementation's natural-log JS lies between
zero and `log(2)` and declares positive smoothing of both normalized laws.
An exactly zero occupied population uses a constant uniform fallback with
zero composition gradient and must still fail the separate occupancy/size
safeguard. If the update is streamed across microbatches, first accumulate
the complete detached counts, obtain one fixed JS count cotangent, then
re-decode under unchanged weights and the same stochastic state to recover
the exact population VJP. Averaging microbatch JS values would define a
different loss.

Expected occupancy and size must be monitored separately. Matching element
fractions alone can coexist with empty or tiny outputs and does not preserve
the size distribution. A size/occupancy safeguard is a declared distribution
prior; it must not be reported as an exact chemistry constraint or repair
distance. The initial continuation adds no charge-changing repair policy.

## Benchmark and readiness gates

The allocated benchmark must use the copied native parent checkpoint,
the supported corpus/split and the exact intended chemistry objective.
Synthetic fixtures establish implementation behavior but cannot establish
native target coverage, whole-training throughput or generation efficacy.

Before starting the continuation, retain evidence for:

- **Representation and search invariants:** occupied slots, atom identities,
  charges and symmetry survive every raw-to-witness transformation; accepted
  targets pass native validation; no extra bond direction or diagonal is
  counted. Failed searches and already-valid points retain their statuses.
- **Gradient contract:** fixed-target loss gives finite useful decoder
  gradients without differentiating search; branch denominators and explicit
  1:3 weights are verified; composition includes all intended outputs.
- **Native coverage and cost:** report raw validity, accepted/censored/
  unsupported fractions separately for real and fresh prior points, actual
  checks and wall time, per-update throughput, host RAM and VRAM. Caps cannot
  silently truncate a nominal 1:3 batch or remove failed points from its mean.
- **Training and continuation fidelity:** restore the authenticated parent,
  train encoder and decoder jointly, retain the same graph/data/split, and
  preserve old evidence. Preserve the parent AdamW moments and RNG lineage
  at the declared objective change, with a separately initialized and
  checkpointed fresh-prior RNG. Record the selected LR/KL policy under the
  new identity. Native continuation/resume tests must verify model,
  optimizer and all relevant RNG state.
- **Retention safeguards:** compare reconstruction, probability-weighted
  sulfur/element/charge content, expected size, raw prior validity and unique
  valid yield on declared validation panels against the copied parent.
  Stop on failed/nonfinite checks; an engineering pass alone is not proof of
  a beneficial chemistry effect.
- **Quota and runtime:** authenticated source/config/data/checkpoint bindings,
  sufficient storage, and a measured feasible segment budget before actual
  submission. Do not inherit the parent's roughly 492 real molecules/second
  as an estimate for a search-heavy 1:3 objective without measurement.

The initial allocated benchmark performs **two full logical updates**
(256 real and 768 fresh prior points per update) on a disposable model copy,
plus a GPU next-update resume probe using four real and 12 prior points.
The latter compares an uninterrupted next AdamW update with serialization,
load and replay, including model/optimizer and parent/fresh-prior RNG state.
It restores the initial native copy afterward; benchmark updates are never
promoted as continuation checkpoints. CPU finite-difference and streamed
population/reconstruction-gradient parity fixtures precede native GPU work.

The native benchmark requires all of the following:

| Gate | Initial limit |
|---|---|
| Mixture and native replay | Exact 1:3 point counts and successful native next-update model/AdamW/RNG replay |
| Fresh-prior repair coverage | At least 1% of scheduled prior points have accepted repair targets; failed/unsupported/already-valid points remain in the denominator |
| Gradients | A measured nonzero repair-only decoder gradient and nonzero encoder/decoder total gradients in every measured update |
| Search/provenance | No `rejected_or_error` attempts; no unreported candidate removal |
| VRAM | Peak reserved CUDA memory below 85% of native device total memory |
| Complete epoch fits | `1.5 * projected_epoch_seconds + 1800 < plan.segment_wall_seconds` (source112: 22h; source115: 71h) |
| Expected size | Between 0.9 and 1.1 times the copied parent on fixed prior coordinates |
| Nonempty fraction | No more than 0.02 absolute decrease from the copied parent |
| Expected sulfur frequency | At most `1.5 * parent_frequency + 0.005` |
| Element and charge JS | Each no more than 0.02 above its copied-parent value |
| Valid and exact reconstruction | Each no more than 0.02 absolute decrease from the copied parent |

The epoch projection uses measured seconds per **real** example times the
1,688,539-example corpus, plus 600 seconds of evaluation allowance. Reporting
all-candidate throughput separately avoids hiding the fourfold point count.
The 1.5 multiplier and 1,800-second reserve are guard margins, not measured
whole-epoch runtime. Benchmark panel size/seed and native allocation receipts
are recorded in the sealed protocol/plan; they cannot be inferred from a
passed synthetic fixture.

The reconstruction thresholds are **two percentage points**, not a 98%
relative-retention rule or an efficacy test. The sulfur threshold is a loose
engineering guard on probability-weighted element frequency across all raw
outputs, not a ban on legitimate sulfur chemistry or a guarantee of useful
molecules. At completed training epochs, the minimum 1% prior coverage guard
counts accepted targets **plus already-valid outputs**, so an improvement
that replaces repaired points with raw-valid points is not treated as lost
guidance. Missing-target points still contribute zero repair NLL.

All acceptance thresholds and coefficients belong in the immutable protocol.
If the bounded search has poor fresh-prior coverage or is too costly, report
the failed gate and retain its evidence. Do not convert that outcome into a
passed continuation by dropping random points, loosening chemistry validation
or silently changing targets.

## Evaluation and endpoint

Use fixed evaluation seeds/cohorts independent of training draws. Compare the
copied parent and chemistry checkpoints on real reconstruction and raw
standard-normal prior generation, both argmax and categorical sampling.
Keep invalid/disconnected attempts in the denominators; report unique valid
yield, failure categories, size and expected/hard element/charge composition.
Any repaired-output panel is separate from raw generation quality.

After every completed chemistry epoch, evaluate a fixed **256-molecule
validation reconstruction panel** and **1,024 fixed raw prior points** for
composition, size, nonempty fraction and raw argmax validity, retaining
the above parent-relative health checks. Every five absolute epochs and
at epoch 100, run the deeper held-out validation panel: 4,096 encoded
validation molecules, 4,096 independent fixed prior points, four categorical
draws per point, 128 local-neighborhood cases at relative scales 0, 0.05,
0.1 and 0.2, and 32 interpolation pairs at alpha 0, 0.25, 0.5, 0.75 and 1.
These monitors preserve the training RNG before evaluation and restore it
afterward, so evaluation cannot change subsequent posterior/prior draws.

Evaluate early in the chemistry phase and at declared milestones without
using the reserved final test for training decisions. Validation-only
monitoring can stop a harmful branch; it cannot establish a replicated or
matched causal improvement. A branch from a current partially trained parent
adds chemistry and random points together, while training time also changes.
The intentionally stopped source098 branch is historical evidence with
declared epoch/data identities. There is no separately completed epoch-100
real-only control and no automatic training-seed replication. Earlier plans
to preserve a continuing control were superseded by the user-directed stop.

At the 100-total-epoch endpoint, authenticate completion and run the full
reserved **20,000-molecule test** for both the frozen plan-bound parent
and the chemistry epoch-100 model, plus the declared deep generation/geometry
panels. The source098 baseline's separate epoch-100/final-evaluation obligation
is superseded by the user-directed stop. Current attempt005 parent versus
chemistry endpoint is confounded by the extra 65 epochs and changed objective;
the fixed-parent test comparison does not isolate the chemistry term.
A submitted controller, queued segment, completed
benchmark or good reconstruction score is not an endpoint result. Historical
Boltz/DeFoG gradient checks remain separate and are not waived by this work.

## Execution record

**Preparation failures retained, 13 October 2026:** source104 attempts001
and002 both failed in metadata-only queue entry **before `bsub`**, so neither
created a job or numerical result. Attempt001 omitted a required strict-JSON
helper; attempt002 included it but passed a string to an authentication API
requiring `Path`. Attempt003 adds the immutable controller `Path` wrapper
and regression fixtures exercising the actual helper, while preserving the
scientific files from attempt002. The [ledger](05_experiment_ledger.md) retains both
archive/manifest/plan identities and the separate local plan-builder and
SFTP-destination recovery events. These are launcher/operator failures,
not evidence against the loss, model or chemistry method; the parent and
unchanged source098 baseline remain preserved.

Source104 attempt003 has these sealed identities:

- [Archive](../outputs/source-releases/source-104-atom-preserving-chemistry-attempt003.tar.gz)
  SHA256 `6e195413af3b9ee395c96b8c6464ef9259653c86a53fa60834a90e0302f742c2`;
  source manifest `65ad28cc056f5578ad9dfbeaa480fddecbb00fad1cbee65994d47b757f1c9e4f`.
- [Plan](../outputs/hpc-submissions/source104a003-chemistry-plan.json)
  SHA256 `8ea4c2279facb67c985f5003c369d821a14f0feb371aef0f74abd75ff73c815f`.
- [Protocol](../configs/chemistry_triangle_1to3.json)
  SHA256 `4df1b796b98bfce8af2975f1c968cf1a83427e20126274ace436a119393b9def`.
- [Preparation record](../outputs/hpc-submissions/source104a003-preparation.json)
  preserves the archived controller/test delta. The controller SHA256 is
  `6c9ed79c33eec607b0b6a49f63fdf15395d8705924fedbf4e8dc60f78a9acfe8`,
  and its fixture SHA256 is
  `479b412204106df54cffd3923d8a455a87771b222be878fb63a58203692513e0`.
- The parent-native evaluator is restored byte-for-byte in the new phase,
  SHA256 `69d7e30784646dd946b3260f2bb614c6e181f611ef475efde7e3ab66f6292a49`;
  the old source098 archive is not changed.

Remote release and campaign are
`/work3/s234818/latentmol/releases/source-104-atom-preserving-chemistry-attempt003`
and `/work3/s234818/latentmol/runs/source104-atom-preserving-chemistry-attempt003`.
Control/plan/submit receipts use
`/work3/s234818/latentmol/submissions/source104-chemistry-continuation-attempt003`.
CPU checks request two cores × 4 GB host RAM / 30 minutes; native benchmark
requests four cores × 16 GB / one exclusive A100 80 GB / two hours; the
promotion callback requests two cores × 4 GB / 15 minutes. Subsequent serial
chemistry segments use 23-hour allocations with a 22-hour internal budget;
the final stage has a separate six-hour allocation. These are requested
resources, not measurements of a completed allocation.

The dedicated fresh-prior seed is **104017** and independent evaluation seed
is **904017**. Initial benchmark population/reconstruction panels each use
256 fixed points/molecules. Benchmark wall cap is 6,300 seconds inside the
two-hour allocation. The controller authenticates readiness/parent/config/
source/data and durable stage outcomes plus quota before any next submission;
`ended(...)` releases a callback but does not approve a failed benchmark.
Failed/unknown outcomes stop, retain the active-phase blocker and evidence,
and require an explicit new sealed recovery attempt rather than a blind retry.

The following table retains the historical attempt001–004 record. Current
attempt005's stopped epoch-35 parent and new plan/receipts are recorded at
the top of this document and in the ledger; no old plan is rewritten.

| Field | Historical attempt001–004 record |
|---|---|
| Parent campaign | source098 large-corpus triangle VAE, segment 002 `20261013T054947Z_vae-train_9e484e3f` |
| Parent completed epoch / updates | Epoch 31 / 204,476 updates; 69 remaining to the 100-total-epoch maximum |
| Copied checkpoint SHA256 | `34fa3dec01ceb140222ad865df938b273c8cde2a2835bad1f6bee2da43e57659` |
| Copied parent configuration SHA256 | `b6704c80c4f3308b5f20623eb0ef21fc39c4b7b1d0cd60d519a9990670b1b254` |
| Dataset / split SHA256 | `54f37815d5901f7d76b855f69e237d97172c72b027e98616f949c08178f9cae3` / `356499265847be6af63f4e70c809ff4b82967fd971b4ffabe94f3aa095adf739` |
| Graph/data scope | Parent canonical 50-slot schema, 1,688,539 supported training molecules, protected split; new binding file pending |
| New source/config/protocol identity | Fixture-only attempt004 archive `e596d67fa1efbedb4590757970de24e30508e8adcd9d82ac45158b6dc69585c4`, manifest `30b4f5c09a1c2008db2b0abee793ff91c672b73fa060eecd55e9a44b29704049`, plan `1a6ff82835a1e45c91727dcd37afe0fb192f61120456754a0ea30578ba047790`; scientific protocol unchanged |
| Training scope | Joint encoder and decoder; real reconstruction/KL plus real chemistry can update encoder; fresh prior branch bypasses encoder |
| Optimizer and RNG scope | Preserve authenticated parent AdamW moments and persisted RNG lineage; checkpoint dedicated new fresh-prior RNG |
| Mixture / microbatch | Logical 256 real to 768 fresh direct standard-normal latent points; microbatch 128; exact 1:3 tail |
| Maximum endpoint | 100 total epochs, including parent-completed epochs |
| Numerical execution | Allocated DTU HPC only |
| Submitted jobs / benchmark result | CPU 29712042 DONE, 112 passed; GPU benchmark 29712043 and callback 29712044 later canceled without execution. Original CPU 29712033 failed; its blocked jobs EXIT without execution |
| Continuation outcome / final evaluation | Pending gates and actual execution |

Each submission, failure, cap, stop, resumed segment and endpoint will be
appended here and in the ledger with exact source/config/data/checkpoint
identities and remote/local artifact paths. Preserve the failed attempt before
retries and distinguish root-reported information from authenticated evidence.
