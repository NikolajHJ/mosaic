# Epoch40 through45: unchanged chemistry training and fresh generation tracking

## Epoch43 running, read-only30 October2026 12:33 CET

The unattended SSH route retrieved [a fresh snapshot](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-1793359995796733500/status.json).
**29849760 RUN** on n-62-18-8; callback29849761 waits normally. Latest persisted
panel12:26:15CET at absolute279256 is2,224/6,596 (**33.72%**) into43, not an
exact live counter or a completed checkpoint. Raw/strict validity772/1,024
(75.39%) versus completed42's782 (76.37%); exact/valid reconstruction256/256,
sulfur1.41499%, zero representation failures in the latest panel. All157 saved
early panels/2,826 individual training-health checks pass. Panel fluctuations
do not establish a completed-epoch trend. Epoch42 remains latest completed.
The existing loss,1:2 ratio, optimizer/anchors, safeguards and serial endpoint45
remain unchanged. The failed local neighbor-loss pilot is not enabled here.
No numerical test, submission or runtime/SSH/VPN setting change in this check.

## Epoch42 completed; epoch43 queued, read-only29 October2026 22:14 CET

The new unattended SSH connector retrieved [the complete source140 snapshot](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-1793308362181258900/status.json).
Segment002 and its callback successfully completed. Authentic checkpoint42 SHA
`5ef851101a5f675edc8f65b31b75c23a5ec460f1c40aa5571829bf8ad6714cfd`
is saved at absolute277032; measured epoch time39,162.440s (**10.88h**),
not the conservative59,011.392s scheduling estimate. All19 end-epoch checks
and134 saved early panels/2,412 checks pass. Fixed-panel raw/strict validity
**782/1,024 (76.37%)**, versus epoch41's741 (72.36%) and40's753 (73.54%).
Exact/valid reconstruction remains256/256; sulfur1.41675%. The
[completed epoch row](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-1793308362181258900/segments/002/20261029T060320Z_chemistry-train_c3f4a7c8/latest_epoch.json)
and [generation report](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-1793308362181258900/generation/epoch0042/metrics.json)
bind these saved results. Additional paired seeds140041/140042 yield argmax
792/811 of1,024 and categorical raw2732/2762 of4,096, versus41's766/757 and
2566/2590. Categorical seed140041 has one representation failure (strict2731).
This is an improvement on these monitoring/evaluation panels, not a full-test
or independent training replication. Argmax valid identities are unique within
each generation panel; bounded fingerprint diversity remains reported separately.

The existing callback automatically submitted **29849760 PEND**, waiting for
job-slot limits on five A100 hosts; **29849761 PEND** normally waits for that
job. Epoch43 has not started. Endpoint45, loss,1:2 ratio and all safeguards
remain unchanged; no new submission or runtime change was made by this check.
No reliable queue start ETA. [Combined read-only summary](../outputs/hpc-results/hpc-run-status-20261029-2214.json).

## Epoch42 about49% through, read-only29 October2026 12:35 CET

[Snapshot014](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-014/status.json)
confirms **29845931 RUN**, callback29845932 PEND normally. Latest panel12:26:01
atabsolute273656 is3,220/6,596 (~49%) into42. Raw/strict validity739/1,024
(72.17%), exact/valid reconstruction256/256, sulfur1.48285%, zero representation
failures in that panel. The preceding panel had764/1,024 (74.61%); these
monitoring counts fluctuate and are not completed epoch42 results. All100 saved
early-health panels/1,800 checks passed. Epoch41 remains latest completed;
no42 checkpoint/end-generation result yet. No science, runtime setting or job
changed in this read-only status collection.

## Epoch42 still running, read-only29 October2026 12:20 CET

[Snapshot013](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-013/status.json)
confirms **29845931 RUN**, callback29845932 PEND on its normal dependency.
Latest saved panel12:16:12 CET is absolute273556,3,120/6,596 (~47%) into42.
Raw/strict validity764/1,024 (74.61%), exact/valid reconstruction256/256,
sulfur1.65681%; all99 saved early-health panels/1,782 individual checks pass.
This is fixed-panel monitoring, not a completed42 checkpoint or a fresh-seed
generation result. Epoch41 remains latest completed, with its previously
documented small generation decline. Endpoint45 and all science/guards remain
unchanged. [Account-wide live queue](../outputs/hpc-results/live-queue-1793272868751196200.json)
at12:21 CET independently confirms this job and the embedding long allocation
RUN. No jobs or runtime settings changed in this read-only check.

## Epoch41 completed; epoch42 running, 29 October 2026 09:46 CET

[Snapshot012](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-012/status.json)
at **09:46:08 CET** records successful GPU29838765 completion at **04:04:40
CET**, followed by the successful serial callback29838766. The
[segment completion](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-012/segment-001-completion.json)
passed with authentic **epoch41**, absolute update **270,436**, checkpoint SHA
`690193d3c0d3cd1059dc4cfec7a9ad4d3d50172a577f25dec0851e91ef50e966`.
The stop reason is the deliberate whole-epoch allocation budget boundary.
The [epoch row](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-012/segments/001/20261028T152712Z_chemistry-train_10e3082e/latest_epoch.json)
measures **41,545.531 seconds (11.54 hours)** for epoch41, processing1,688,539
real molecules and3,377,078 fresh-prior points. The completion record's
59,011.392 seconds is the conservative scheduling estimate, not measured time.
All **19 end-of-epoch checks passed**.

| Completed checkpoint: fixed monitoring panel | Epoch40 | Epoch41 |
| --- | --- | --- |
| Raw / strict prior validity | 753 / 1,024 (73.54%) | 741 / 1,024 (72.36%) |
| Exact / chemically valid reconstruction | 256 / 256 for both | 256 / 256 for both |
| Sulfur frequency | 1.4394% | 1.52873% |

The completed epoch41 validity is **12 fewer valid graphs, or1.17 percentage
points below epoch40** on this fixed panel. This is a small deterioration,
not continued monotonic improvement; reconstruction is retained and all
composition/health safeguards pass. The [saved-checkpoint observation receipt](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-012/generation-segment-001.json)
passed integrity checks and binds the new
[epoch41 generation report](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-012/generation/epoch0041/metrics.json)
against the [epoch40 report](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-012/generation/epoch0040/metrics.json).
These two additional prior panels, using seeds140041/140042, are distinct from
the training monitoring panel:

| Paired generation panel | Epoch40 raw valid | Epoch41 raw valid |
| --- | --- | --- |
| Seed140041, argmax | 774 / 1,024 (75.59%) | 766 / 1,024 (74.80%) |
| Seed140042, argmax | 773 / 1,024 (75.49%) | 757 / 1,024 (73.93%) |
| Seed140041, four categorical draws per latent | 2,676 / 4,096 (65.33%) | 2,566 / 4,096 (62.65%) |
| Seed140042, four categorical draws per latent | 2,599 / 4,096 (63.45%) | 2,590 / 4,096 (63.23%) |

Thus neither of these additional panels establishes an epoch41 generation
improvement. Argmax valid identities are unique within each panel. Mean
pairwise fingerprint Tanimoto on the declared256-identity cohorts changed
0.1182 to0.1159 and0.1199 to0.1186, respectively; no collapse in that particular
diversity statistic is apparent, but it is a limited cohort measure. The
epoch41 seed140041 categorical panel contains **one representation failure**,
so its strict validity is2,565 rather than2,566; epoch40 seed140042 similarly
had2,598 strict versus2,599 raw. Invalid outputs remain represented in the
saved report. Four draws sharing a latent are correlated, and these panels
are paired across checkpoints, not independent replications or a full-test
estimate.

The callback automatically submitted **GPU29845931**, which is **RUN** on
`n-62-18-8` since **07:02:21 CET**, with callback **29845932 PEND** normally.
Its [actual training record](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-012/segments/002/20261029T060320Z_chemistry-train_c3f4a7c8/run.json)
restores the exact completed41 checkpoint. The latest persisted monitoring
panel at **09:37:50 CET**, absolute update **271,956**, is1,520 of6,596 updates
into epoch42 (approximately **23.04%**): raw/strict validity **750 / 1,024
(73.24%)**, exact/valid reconstruction256/256, sulfur1.45131%, zero
representation failures in that panel. Across this campaign's **83 saved
early-health panels** (67 in epoch41 and16 so far in epoch42), **all1,494
checks passed**. Earlier representation failures remain recorded. Epoch41 is
the latest completed checkpoint; this epoch42 panel is not a checkpoint or
exact live update count. Loss,1:2 ratio, optimizer/RNG policy, safeguards and
endpoint45 remain unchanged. This review used saved local artifacts only;
no runtime code, settings or jobs changed.

## Actual training running, 28 October 2026 20:22 CET

[Snapshot011](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-011/status.json)
at **20:22:52 CET** confirms GPU **29838765 RUN** on `n-62-18-9`, with the
allocation started at **16:20:45 CET**. The current-release
[CUDA gate](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-011/training-launch-cuda-gate.json)
and [JUnit](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-011/training-launch-cuda-pytest.xml)
record **649 passed, zero failures/errors/skips**, in 104.727 seconds. The
[epoch40 handoff](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-011/epoch40-handoff.json)
passed all six checks: exact epoch/update, model, optimizer, native/fresh RNG
and frozen epoch36 reference. It binds the authentic epoch40 checkpoint SHA
`31af05597dbd176b02a8108651db0d6b33bb3a8883f69ecf354a4465ed18787e`.

The [actual training record](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-011/segments/001/20261028T152712Z_chemistry-train_10e3082e/run.json)
was created at **16:27:12 CET** and entered training at **16:27:15 CET** with
that exact resume hash. This establishes successful passage through the
corrected observer and the observer-to-trainer process handoff. The snapshot
contains **22 saved early-health panels; all 396 checks passed** (18 per
panel). The latest
[saved panel](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-011/segments/001/20261028T152712Z_chemistry-train_10e3082e/early_health/update_000265956.json)
was written at **20:14:23 CET**, absolute update **265,956**: 2,116 of 6,596
updates into epoch41, approximately **32.08%**. This is a persisted observation,
not an exact live update count or a completed epoch41 checkpoint.

| Latest fixed monitoring panel | Result |
| --- | --- |
| Raw / strict prior validity | 754 / 1,024 for both (73.63%) |
| Exact / chemically valid reconstruction | 256 / 256 for both |
| Sulfur frequency | 1.42309% |
| Representation failures in latest panel | 0 |

The fixed-panel raw validity has fluctuated between **689 and 790 of1,024**
within this segment. The preceding panel at update265,856 had **one
representation failure** (raw761/strict760), so zero failures must not be
claimed across all panels. Every saved reconstruction panel remains256/256.
These are repeated fixed monitoring panels, not independent seeds or a full
test-set estimate. **Original epoch40 remains the latest completed checkpoint**;
epoch41 is still in progress. Callback **29838766 PEND** waits normally for
the running allocation. Loss,1:2 ratio, optimizer/RNG policy, guards and
endpoint45 remain unchanged. This status review used saved local artifacts
only and changed no runtime code, settings or jobs.

## Read-only status, 28 October 2026 13:57 CET

[Snapshot010](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-010/status.json)
at13:57:59 CET confirms GPU **29838765 PEND**. Its reservation has moved to
12 of16 CPU slots on `n-62-18-8`; insufficient processor units for the affinity
request block execution on one host. All14,944 pending seconds are eligible.
The scheduler currently says **no sooner than28 October21:19:19 CET**. That is
a provisional scheduling estimate, not a guaranteed start time. Callback
**29838766** waits normally. No GPU gate, ownership probe, actual segment or
epoch41 exists. Completed CPU/callback records have aged out of live scheduler
history; saved DONE logs and the2,041 CPU passes/profile/adoption/CLI checks
remain valid. This read-only check changed no runtime code, settings or jobs.

## Read-only status, 28 October 2026 10:39 CET

[Snapshot008](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-008/status.json)
at 10:39:51 CET confirms GPU **29838765 remains PEND**, with all 16 CPU
slots reserved on `n-62-18-11` and job-slot limits on four hosts. Its pending
time is eligible (3,056 seconds); the serial callback **29838766** waits on
its expected `ended(29838765)` dependency. Reservation is not execution.
No current CUDA/ownership result, segment directory, training update or
epoch41 exists. The previously passed CPU/profile/adoption/actual-command
checks remain valid. No reliable start ETA is established; this read-only
collection changed no code, settings or jobs.

## Correction CPU validated and resubmitted; GPU pending, 28 October 2026 09:50 CET

New immutable **`source140-epoch40-to45-continuation-attempt002`** preserves the
failed attempt001 and resumes the same exact epoch40 checkpoint through45.
The [combined submission summary](../outputs/hpc-results/fix-resubmit-20261028-summary.json)
and [snapshot007 at 09:50:38 CET](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-007/status.json)
record **CPU 29838695 DONE**, **after-CPU 29838696 DONE**, and new
**GPU 29838765 / serial callback 29838766 PEND**. The GPU waits on job-slot limits
on five hosts; its callback waits on the expected dependency. There is no
reliable start ETA, no new CUDA or ownership-probe result, no fresh generation
observation and no epoch41 training update yet.

The full allocated CPU suite passed **2,041 fixtures**, with **120 CUDA-only
skips**, zero errors/failures, in 302.19s JUnit walltime. The full CPU stage was
478s; its callback was 149s. The repair profile and authenticated source135/
source138 adoption also passed. The callback's
[actual generated CLI dispatch](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt002/snapshot-007/launch-command-001.json)
passed with the exact epoch40 resume hash and endpoint45, without model setup
or updates. The profile's inherited text mentioning 1:3 describes its diagnostic
four-real/32-prior cohort; **actual training remains 256 real/512 prior, 1:2**.

| Attempt002 identity or receipt | SHA256 |
| --- | --- |
| Source archive | `375272634d1836625e3d9a2bb8548e93f4cf4ac982ad1853fd0c621af8a344ef` |
| Scientific manifest | `ac6b06edeff9ce6d35413d84890ee3738a68f517230ac06bd6345f6f86ca4d65` |
| Plan | `a794d3f7a9b3bcbcdc357aadb6feade06806a4692b08a65e7c3a0e8e0a9420f6` |
| CPU gate | `9639beecfa36ac87b2223dc3d3a352ca562d690103aafb59cc40e05080b3a973` |
| Repair profile | `2b7684c16180a356c7fb4d49f78bcf3077168089d1d3765a457a8f25f3f37172` |
| Evidence adoption | `142cf587de725c7c7c8cc9ed302bc9068be2a2de255b3ee5a147a9a6a4cda278` |
| Actual CLI dispatch | `0f0cd3224f0f9de4584c99a187ac190f952b434f97c9ed1220bc29523d37d0bc` |

The [seal/preparation](../outputs/hpc-submissions/source140-epoch40-to45-continuation-attempt002-preparation.json)
binds 657 archive members / 585 scientific members. Its historical
`sealed_not_submitted` status is superseded by the actual job receipts in
snapshot007. The
[byte/AST scope audit](../outputs/implementation-checks/source140-attempt002-process-fix-static.json)
confirms identical scientific plan and loss/trainer/runtime bytes, with the
failed source140 release preserved. Source138's completed checkpoint and all
previous failed evidence remain unchanged. A CPU readiness pass or queued GPU
is not a new native handoff, corrected GPU execution or training start.

### Path interface and exclusive-process ownership mechanism

The path fix normalizes `checkpoint = Path(checkpoint)` inside
[`observe_checkpoint`](../src/latentmol/chemistry_epoch45_generation.py), so
both initial string constants and saved-epoch string paths reach the same
hash/identity checks. The campaign's observation-receipt hashing boundary is
also normalized. New regressions exercise real observer bodies for initial40
and saved41 with string/`Path` arguments, persisted receipts and tamper checks,
plus actual initial/saved worker phase functions. They do not replace the
observer with a mocked boundary.

Inspection also found a second process-lifetime problem: the old worker would
initialize CUDA for its baseline observer, delete tensors and call
`empty_cache`, then launch training in a different process while the original
process still existed. The same retained-context pattern is present in the
failed embedding native-prediction path. NVIDIA's [CUDA12.1 guide, section 6.4](https://docs.nvidia.com/cuda/archive/12.1.0/pdf/CUDA_C_Programming_Guide.pdf)
restricts exclusive-process devices to one context across processes. PyTorch's
[`empty_cache` documentation](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.empty_cache.html)
describes releasing unused allocator memory; it does not describe ending the
owning process's CUDA context. The recorded native error and this launch
structure identify a concrete ownership mechanism. The new allocated GPU
probes still have to demonstrate that the corrected handoff works on DTU.

[`cuda_process.py`](../src/latentmol/cuda_process.py) now provides the shared
CUDA-free coordinator check, bounded child execution, immutable phase logs,
and cleanup of only its own child process group, including surviving workers.
It does not initialize CUDA when checking the parent. In
[`chemistry_epoch45_worker.py`](../scripts/hpc/chemistry_epoch45_worker.py),
the coordinator performs metadata work only. The CUDA pytest child exits,
then two separate tiny CUDA ownership probes must pass. Baseline observation,
the unchanged trainer command, and each saved-epoch observation run as serial
CUDA child owners; each exits before the next begins. Child failure, timeout,
identity mismatch or exhausted shared deadline prevents advancement.

The original loss, sampling, references, moments, quarter-step policy and all
native/fresh RNG checks remain unchanged. Full current-release CUDA coverage,
exact40 state handoff and raw observations remain required before training.
The 23-hour shared deadline, 30-minute observation reserve, whole-epoch fit,
health checks and serial 24-hour allocations are retained. The submitted GPU
requests one exclusive 80GB A100 / 16 CPU slots; the chain still stops at45 and
saves every epoch. No GPU result is established by these CPU checks.

## Observer correction and resubmission authorized, 28 October 2026

The user requested **“please fix and resubmit”** after the terminal failures
below. The VAE correction is scoped to the checkpoint-path interface used by
the additive generation observer, with production-boundary regression coverage
for both string and `Path` inputs. The failed source140 release, jobs, logs and
receipts remain preserved. A corrected immutable release must pass the relevant
allocated launch checks before its continuation can begin.

The continuation still resumes the exact completed epoch40 checkpoint through
absolute epoch45. Loss, 1:2 real-to-prior sampling, frozen original35/original36
references, optimizer moments and displacement policy, RNG state, observations,
all scientific and health safeguards, saved epochs, and serial allocation
limits remain unchanged. This authorization is **not** a correction pass,
replacement submission or training start; exact release identities and check
receipts will be recorded when available.

## Terminal startup failure after CUDA passed, collected 28 October2026 09:08 CET

[Read-only snapshot009](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt001/snapshot-009/status.json)
supersedes the queued status below. GPU **29829875** ran **02:44:35--02:49:37 CET**
on n-62-18-11 and exited1. All **631 current-release CUDA fixtures passed**, with
zero failures/errors/skips in96.51s. The
[CUDA gate](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt001/snapshot-009/training-launch-cuda-gate.json)
SHA is `78cd261dd04e29d09a6c13430ac780f67272743dab149b1e6b1217148271f36f`.

Production setup reached the new baseline observation before any trainer
subprocess or update. `chemistry_epoch45_worker.py` passes the `CHECKPOINT`
string to `observe_checkpoint`, whose hash call expects a `Path` and invokes
`.open("rb")`. It raised **`AttributeError: 'str' object has no attribute 'open'`**
at `chemistry_epoch45_generation.py:74`. This is an observer interface error;
the log does not show a missing checkpoint or a chemistry-loss failure.
The successful handoff receipt was not written, because it is persisted after
the observation. No epoch41 training segment or completed checkpoint exists.
Original epoch40 remains the latest completed checkpoint.

Callback **29829876** exited02:49:41 on the absent segment-completion receipt
and submitted no successor. Aged-out scheduler records are superseded by saved
terminal LSF logs and the allocation's failed run record. The snapshot records
this failure; no runtime code, loss, safeguard or job was changed by collection,
and no replacement release or submission is established by this status check.

## Latest: CPU passed; GPU queued, 27 October2026 12:18 CET

[Snapshot008](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt001/snapshot-008/status.json)
and [submission summary001](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt001/submission-summary001.json)
confirm **CPU29829798 DONE** and **callback29829799 DONE**. Full fixtures:
**2,028 passed,115 CUDA-only skips,zero failures/errors**,298.257s JUnit wall;
the full CPU stage completed successfully in470s. The original reference/
optimized repair profile passed, as did source135/source138 evidence adoption.
This is authenticated historical scientific evidence reuse, not a new native
or2,500-update proof.

The allocated callback's [actual CLI dispatch check](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt001/snapshot-008/launch-command-001.json)
passed at12:18:07, without model setup or updates. Its actual command contains
the exact source138 epoch40 checkpoint/SHA and endpoint45. The harness's
`start_epoch=36` field names the preserved original handoff/reference ancestry;
the explicit `resume` pair and callback's `completed_epoch=40` identify the
operational training start. No origin/reference was changed to40.

The callback submitted **GPU29829875** plus serial **callback29829876**.
At12:18:50 both are PEND: GPU waits for **job-slot limits on five hosts**,
callback waits on its expected dependency. Request: one exclusive80GB A100,
16CPU slots,24h LSF,23h internal. **No current-release CUDA gate, exact native
epoch40 handoff, fresh generation observation or epoch41 training has run yet.**
There is no reliable queue start ETA. All these GPU requirements remain
mandatory before actual updates. No independent study or completed source138
release/checkpoint was changed.

Authenticated receipt hashes:

```text
CPU gate      9982ebf0a87d8c7f68ef8be9f1f86a1501c8ebe4745bef033793258abd38de26
CPU JUnit     4be28b8faee60f34f397d187c69130c8075c36ec529f531acafbfb3c3da8093b
Repair profile f25d127cc32b0b03c3fc5f5a83c76b00b8fd3e677640cf5e03f1021590e0347f
Adoption      659beb30bd0e411080687784881189e33ff272f49f274c37662887f87fd20e75
Actual argv   be8f897df4771984c5b223458105a4d5833535bcc2287ad7edd7955c48b74c31
```

The user authorized **five additional epochs,41--45**, on27 October2026,
conditional on passed checks, with the loss and safeguards unchanged, every
epoch saved, and validity/diversity tracked. This is a **new immutable campaign**;
the successfully completed [original36->40 campaign](75_frozen_reference_epoch40_continuation.md)
is not edited or restarted. Packaging/submission is not a passed gate or actual
training. Numerical checks execute only inside allocated DTU jobs.

## Exact state and scientific scope

Training resumes from source138's completed epoch40 checkpoint:

```text
/work3/s234818/latentmol/runs/source138-vae-cli-startup-fix-attempt001/segments/004/20261026T230230Z_chemistry-train_e9f28b40/epoch_0040.pt
SHA256 31af05597dbd176b02a8108651db0d6b33bb3a8883f69ecf354a4465ed18787e
epoch40, absolute263840 updates
```

The **training state is epoch40**, but frozen references stay **original35 for
occupancy and the original repair ramp**, and **authentic original36 for same-z
valid-target NLL**. Replacing either anchor with epoch40 would change the loss.
All source138 protocol/data/configuration fields remain identical:256 real/512
prior points (1:2), full atom-preserving repair NLL, frozen valid-target weight0.1,
all-attempt means, composition weights, search budgets, FP32/microbatch128,
12workers/four Torch threads/16CPU slots, quarter actual native AdamW displacement
with inherited moments, and both native and fresh-prior RNG. No scheduler reset,
new random training points policy or new objective is introduced.

The distinct `epoch45_extension` marker binds the exact parent plan, checkpoint,
completion and branch SHA values; the historical `valid_reference_continuation`
marker remains original36->40. The operational endpoint reader recognizes45
only with the complete explicit new marker. Epoch46 and the global100 endpoint
are not authorized. Every new epoch checkpoint persists the extension marker
and is retained.

## Authentication and execution

1. The read-only preflight hashes original35, original36, epoch40, and all source135
   proof inputs; verifies completed source138 jobs and official user quota.
2. The new release is built from the sealed source138 archive plus a closed list
   of metadata/observation changes. Every model/loss/search/update/configuration
   executable remains byte-identical. Shared metadata modules are AST-audited:
   constants/function coverage and all other function bodies remain unchanged.
3. Full allocated CPU fixtures and the existing reference/optimized repair
   profile run. Additive adoption authenticates the original source135 scientific
   proof through source138's exact original receipts, all268 early panels and
   epoch40 end health. Old receipts/failed evidence remain unchanged.
4. The allocated callback runs the **actual generated CLI command**, including
   the epoch40 resume pair, stopping before model setup. All existing source,
   readiness, stability, quota, predecessor and serial-launch checks remain.
5. The first GPU allocation runs the complete original mandatory CUDA suite plus
   extension fixtures, with the same zero-failure/error/skip requirement. Actual
   production setup then builds original35/original36 references before restoring
   epoch40, and checks exact model, optimizer and all native/fresh RNG state.
6. A fresh generation baseline is saved for epoch40 before training. The original
   trainer runs unchanged loss/update bodies. All32/every100 early checks,19
   end-epoch checks, whole-epoch budgeting, per-update deadlines and exact resume
   checks remain mandatory. Source138's old algorithm/source identities apply
   only to the exact initial40 handoff; subsequent checkpoints require the new
   exact release/plan/algorithm/extension identities.
7. Every saved epoch receives fresh-panel reports before the serial callback
   may advance. Allocations request at most24h, with a shared23h worker budget
   and observation reserve; whole-epoch fit uses the unchanged conservative
   runtime amendment. Each callback authenticates completion, health, checkpoint,
   optimizer evidence, observation integrity and quota. The chain stops at45.

## Validity and diversity reports

The original monitoring panel and its thresholds are unchanged. New observations
use two freshly declared fixed CPU latent seeds,140041/140042,1,024 standard
Gaussian latent vectors each, batch32, and four categorical draws per latent.
The same new coordinates/draw seeds are used at40 and each41--45 endpoint.
This gives2,048 argmax and8,192 categorical attempts per checkpoint. These are
paired checkpoint comparisons; categorical draws within one latent are correlated.
They are not an independent full-test or binding study.

Saved reports include raw/strict validity, separate chemical/representation
failures, unique valid count/yield, duplicate frequency implicit in these counts,
atom-count/composition distributions, atom entropy, fingerprint diversity and
unique Murcko scaffolds. Fingerprint internal diversity is **1 minus mean
pairwise Morgan Tanimoto**, so larger means more structural diversity. The new
structural report uses at most256 unique raw-valid identities ordered by
SHA256(identity), without score selection. It is a bounded distinct-identity
diagnostic, not frequency-weighted full-population diversity. Scaffold counts
explicitly include the empty scaffold for acyclic molecules. The inherited
evaluator's lexicographic256-cohort similarity is also retained with its original
label; it is not silently replaced by the new hash-cohort report.

No graph is repaired, filtered or dropped for being invalid/duplicate. Gzipped
raw labels/edges are retained. Each report binds exact checkpoint/model/epoch/
update, plan/source, latent/sample seeds and raw-output hashes. The observer
preserves every model mode and native/CUDA/named-generator RNG state and checks
the model digest. New observations do not contribute gradients or choose which
checkpoint to resume; an observer integrity error stops automatic advancement.
No new scientific metric threshold is substituted for existing safeguards.

## Relevant code

- [`chemistry_epoch45_extension.py`](../src/latentmol/chemistry_epoch45_extension.py):
  closed scope, source138/proof authentication, strict metadata source audit,
  initial40 versus exact later-resume identity.
- [`chemistry_continuation.py`](../src/latentmol/chemistry_continuation.py): existing
  full setup/checkpoint/update/health/runtime code; only metadata resume identity,
  persisted extension marker and retention integration change.
- [`chemistry_epoch45_campaign.py`](../scripts/hpc/chemistry_epoch45_campaign.py):
  inherited complete CPU/CUDA and serial campaign plus observation integrity.
- [`chemistry_epoch45_worker.py`](../scripts/hpc/chemistry_epoch45_worker.py): exact
  native initial handoff, baseline, actual trainer command and saved-epoch reports.
- [`chemistry_epoch45_generation.py`](../src/latentmol/chemistry_epoch45_generation.py):
  fresh raw observations, native failure separation and structural diversity.
- [`large_vae_evaluation.py`](../src/latentmol/large_vae_evaluation.py): unchanged
  raw argmax/categorical evaluator and inherited metrics.
- [`test_chemistry_epoch45_extension.py`](../tests/test_chemistry_epoch45_extension.py)
  and [`test_chemistry_epoch45_generation.py`](../tests/test_chemistry_epoch45_generation.py):
  allocated tamper/endpoint/worker/source/RNG/output contracts.

## Preparation status

Read-only preflight001 succeeded27 October2026 at12:03:39 CET. Both epoch40 jobs
were completed/inactive and all declared protected hashes matched. Work3 quota
was46.07/100GiB used (~53.93GiB free). These are metadata/storage observations,
not new numerical passes. The new implementation is untested until its allocated
CPU and GPU gates pass. Immutable identities and actual job receipts will be
recorded below after packaging/submission.

## Immutable attempt001 submitted; CPU running, 27 October2026 12:08 CET

The [sealed preparation](../outputs/hpc-submissions/source140-epoch40-to45-continuation-attempt001-preparation.json)
contains655 archive members/583 scientific members, built from source138 plus
eleven declared metadata/observation overlays. The static AST/byte audit passed
without changing any model/loss/search/update/configuration body. Identities:

```text
Archive  3fd30b055b078a44c247779966e6c353cf6b66279b9740655e70294766c695f8
Manifest 514a5c5c521510eb09556861542d8733ba91b13afb3a7491c397fec58bd0ba86
Plan     934764491608b7f354bf5d47aec0f72f014a66d845f3c3ff76589d71c2ee3fb8
```

[Stage and queue001](../outputs/hpc-submissions/source140-epoch40-to45-continuation-attempt001-stage-and-queue-001-output.json)
both returned zero and submitted **CPU29829798/callback29829799**.
[Read-only snapshot002](../outputs/hpc-results/source140-epoch40-to45-continuation-attempt001/snapshot-002/status.json)
at12:08:41 CET confirms CPU RUN on n-62-11-66, full fixtures in progress,
callback PEND on its expected dependency. **No new CPU pass, GPU job, CUDA pass,
exact40 handoff pass, fresh-panel result or epoch41 training start exists yet.**
The callback will submit the serial GPU workflow only after full CPU/profile/
adoption and actual-argv checks pass; its first allocation retains CUDA and
native handoff checks before any update. Submission is not an experiment result.
No historical release/checkpoint or independent study was changed.
