# Large-corpus triangle VAE pipeline

**Epoch-35 chemistry readiness, 13 October 2026 23:00 DTU-local:** new
plan005 CPU 29712275 completed **112 passes in 13.02 seconds**; native
benchmark 29712276 and promotion callback 29712277 remain PEND. No full
chemistry training has started. The original real-only branch remains
intentionally stopped at published epoch 35, with partial epoch 36 discarded.
The [verified capsule](../outputs/hpc-results/source104a005-first-suite-001.tar.gz)
and [docs53](53_atom_preserving_triangle_continuation.md) retain the new
parent-bound fixture pass and remaining gates toward 100 total epochs.

**Epoch-35 chemistry suite submitted, 13 October 2026:** after intentional
source098 interruption, attempt005 has CPU 29712275, native benchmark
29712276 and promotion callback 29712277. New plan-bound readiness is still
pending; full chemistry training has not yet started. The preserved immutable
epoch-35 parent leaves 65 epochs to the new branch's total-100 maximum.
Scientific source/protocol/data/configuration are unchanged, with a new
checkpoint/campaign/plan identity. [Docs53](53_atom_preserving_triangle_continuation.md)
and the [ledger](05_experiment_ledger.md) retain scope and receipts.

**Real-only endpoint intentionally interrupted, 13 October 2026 22:54
DTU-local:** actual EXIT receipts establish the user-directed stop of
source098 training 29703164 and its automatic callback 29703165. The latest
fully published point is **epoch 35 / 230,860 updates**, with partial
epoch 36 discarded. The original raw run JSON remains historically
`running` after termination; it is not rewritten as completed or unstable.
All source098/original checkpoints and evidence remain preserved, while the
old baseline-to-100/final-evaluation obligation is superseded. New chemistry
attempt005 copies immutable `epoch_0035.pt`, targets 100 total epochs
(65 remaining), and retains native readiness gates. Its submission is
pending at this entry. [Stop receipt](../outputs/hpc-submissions/source104a005-parent-snapshot.json)
and [docs53](53_atom_preserving_triangle_continuation.md) identify the
changed-objective continuation; no 100-epoch real-only control is claimed.

**Real-only baseline stop authorized, 13 October 2026:** the user requests
stopping source098 training and its automatic continuation and using its
most recent completed authenticated checkpoint for the new 1:3 chemistry
phase toward 100 total epochs. The previous real-only 100-epoch continuation
and final-evaluation obligation is superseded; this baseline will be recorded
as intentionally interrupted, not completed or scientifically unstable.
Actual stop/cancellation receipts and final saved epoch are still pending.
All source098/original checkpoints and evidence remain preserved. The
[ledger](05_experiment_ledger.md) and [docs53](53_atom_preserving_triangle_continuation.md)
retain the separate changed-objective branch and its unchanged native gates.

**Separate chemistry readiness, 13 October 2026 20:05 DTU-local:** source104
attempt004's allocated **CPU 29712042 DONE**, **112 passed in 12.59 seconds**.
The native benchmark **29712043 PEND** and promotion callback **29712044 PEND**
remain unexecuted; no full chemistry continuation is submitted. The
[verified capsule](../outputs/hpc-results/source104a004-first-suite-001.tar.gz)
establishes fixture readiness, not native repair coverage/runtime or generation
benefit. Attempt003's fixture failure and unexecuted canceled jobs remain in
the [ledger](05_experiment_ledger.md). Source098's original training objective,
running campaign, checkpoint lineage and final-evaluation obligation remain
unchanged. [Docs53](53_atom_preserving_triangle_continuation.md) explains the
separate new 1:3 chemistry branch and remaining promotion gates.

**Separate chemistry branch, 13 October 2026:** the user-authorized source104
attempt003 suite is submitted for CPU fixtures, native benchmarking and a
gated callback. It copies the completed epoch-31 source098 parent and adds
1:3 real/fresh-prior training with atom-preserving repair-target NLL and
composition safeguards, targeting 100 total epochs if its gates pass.
Attempt003 CPU gate subsequently failed at one malformed-storage fixture
(111 passed / one failed); blocked native/callback jobs have not run and are
being canceled while a fixture-only recovery is prepared. This does not
alter this source098 campaign's reconstruction-plus-KL objective,
data, checkpoints or final-evaluation obligation. No native pretrained-GPU
chemistry benchmark or full training submission is established yet.
[Docs53](53_atom_preserving_triangle_continuation.md) and the
[ledger](05_experiment_ledger.md) retain exact scope and submitted identities.

**Current training status, 13 October 2026 18:39 DTU-local:** segment 002
**29703164 RUN**, **31/100 epochs completed / 204,476 updates**. The
[running record](../outputs/hpc-results/source100-first-suite-005/runs/source098-large-triangle-evaluator-recovery/segments/002/20261013T054947Z_vae-train_9e484e3f/run.json)
retains the same source098 archive, configuration
`538eff49a8bed9326ef6be4fbcdb028be4e2334bc5a32c85a00e422a2fbe0610`,
data/split and authenticated epoch-20 resume checkpoint
`03596e7e32eeece110adf6e1538ac1617628436ff5b23a10332dd260bf6118ee`.
Callback **29703165 PEND** awaits `ended(29703164)`. Joint encoder/decoder
training remains reconstruction plus KL only, with no extra prior training
points or chemistry/repair loss. No scientific setting or submission changed
during this status check.

Epoch-31 [telemetry](../outputs/hpc-results/source100-first-suite-005/runs/source098-large-triangle-evaluator-recovery/segments/002/20261013T054947Z_vae-train_9e484e3f/latest_epoch.json)
has 491.79 real molecules/second, about 57.3 minutes per epoch, no clipped
updates, and routine chemistry valid/exact 256/256. The latest expanded
panel, at epoch 30, has **4,090 valid / 4,084 exact of 4,096** (99.854% /
99.707%), compared with epoch 20's 4,087 / 4,077 on the same panel.
This modest reconstruction improvement is not a generation or final-test
result.

The epoch-25 deep panel in the [persisted metrics](../outputs/hpc-results/source100-first-suite-005/runs/source098-large-triangle-evaluator-recovery/segments/002/20261013T054947Z_vae-train_9e484e3f/metrics.jsonl)
now separates real-molecule latent decoding from unconditional-prior
decoding. The pinned evaluator draws 4,096 direct standard-normal
`N(0,I)` latent points in 256 dimensions with a fixed seed, without
rescaling or whitening. Encoded validation posterior means yield **4,088/4,096 valid
argmax graphs (99.805%)** and **16,326/16,384 valid categorical graphs
(99.646%)**. Unconditional prior points yield **47/4,096 valid argmax
graphs (1.147%; 47 unique)** and **73/16,384 valid categorical graphs
(0.446%; 70 unique)**. All raw attempts are counted, including invalid
or disconnected outputs; no repair, rejection resampling or Boltz scoring
is used. Thus high reconstruction accuracy does not yet translate into
reliable raw generation from random prior points. These results alone do
not identify the cause, compare causally against the historical architecture,
or establish a need for changed training. The authorized training continues
unchanged; the reserved final test remains unopened.

The predefined small geometry panels show **127/128 valid argmax outputs**
at each relative neighborhood scale 0, 0.05, 0.1 and 0.2. Interpolation
midpoints have **1/32 valid argmax outputs** and **3/128 valid categorical
outputs**, while endpoint argmax validity is 32/32 and 31/32. These small
panels distinguish local robustness from global latent-space consistency;
they do not establish semantic interpolation or a VAE validity guarantee.

[Verified capsule005](../outputs/hpc-results/source100-first-suite-005.tar.gz),
SHA256 `1bfde060b3f5889a2ff41532feec9c78441b16820c4b7c370236da2fbd4754a9`,
has 94 safe members / 93 verified files, exported
`2026-10-13T16:39:32.206520+00:00`. Local work read only authenticated
JSON/log bytes with Python 3.12.14; no local scientific numerical work ran.
The independent contact-boundary observer remains queued without a new
GPU result, as recorded in [docs51](51_defog_matched_pilot.md).

**Historical training status, 13 October 2026 09:40 DTU-local:** segment 002
**29703164 RUN**, with **21/100 epochs completed**, 138,516 updates. The
[running record](../outputs/hpc-results/source100-first-suite-004/runs/source098-large-triangle-evaluator-recovery/segments/002/20261013T054947Z_vae-train_9e484e3f/run.json)
authenticates resume from epoch 20 and parent `last.pt` SHA256
`03596e7e32eeece110adf6e1538ac1617628436ff5b23a10332dd260bf6118ee`,
unchanged configuration `538eff49a8bed9326ef6be4fbcdb028be4e2334bc5a32c85a00e422a2fbe0610`,
source098, data and split. It started 05:49:48 UTC on `n-62-18-9`, A100 80 GB,
batch 256. Callback **29703165 PEND** waits on `ended(29703164)`. The automatic
handoff has now reached numerical resume; neither the segment endpoint nor
100 epochs is completed.

Epoch-21 [telemetry](../outputs/hpc-results/source100-first-suite-004/runs/source098-large-triangle-evaluator-recovery/segments/002/20261013T054947Z_vae-train_9e484e3f/latest_epoch.json)
has routine valid/exact 256/256, reconstruction loss 0.0001486911 on the
4,096-example validation cohort, no clipped updates, gradient norm maximum
1.40569 and 490.02 real molecules/second, about 57.5 minutes per epoch.
The latest expanded chemical reconstruction panel remains epoch 20:
4,087 valid / 4,077 exact of 4,096. The smaller routine chemistry panel is
rotating and cannot replace that expanded panel. Neither is unconditional
generation or the reserved final test; deep prior evaluation awaits epoch 25.

[Verified capsule004](../outputs/hpc-results/source100-first-suite-004.tar.gz),
SHA256 `1a8ebaf18401ddf7fe22a7c01241ae61c335f9be20008be4860f7dc44d4fe9c1`,
94 safe members / 93 verified files, exported 07:39:53.180908 UTC, retains
the running segment and scheduler snapshot. Campaign `progress.json` still
describes completed segment 001 at epoch 20; the running segment's latest
epoch record establishes epoch 21. Official work3 usage is 87.79 / 100 GiB
at this read. The root made no submissions or scheduler/scientific mutations
in this status turn. Independent native affinity and structural diagnostics
both completed; their localization results and remaining failed-guard limits
are in [docs51](51_defog_matched_pilot.md) and the [ledger](05_experiment_ledger.md).

**Historical training status, 12 October 2026 19:48 DTU-local:** **20/100 epochs
completed**, 131,920 updates, under unchanged source098 scientific/data/config
identities. Segment 001 stopped normally with `wall_budget_epoch_boundary`,
return code 0 and a passed [durable completion](../outputs/hpc-results/source100-first-suite-003/runs/source098-large-triangle-evaluator-recovery/stage-completions/segment-001.json).
The [controller decision](../outputs/hpc-results/source100-first-suite-003/submissions/source100-triangle-auto-continuation-attempt001/decisions/segment-001/decision.json)
at 15:44:26 UTC authenticated `last.pt` SHA256
`03596e7e32eeece110adf6e1538ac1617628436ff5b23a10332dd260bf6118ee`
and submitted **segment 002 29703164 PEND** plus **callback 29703165 PEND**,
`ended(29703164)`. The first automatic scheduling handoff is verified;
there is no current scientific RUN or completed second-segment resume.

Epoch 20's same 4,096-molecule expanded validation panel has **4,087 valid
(99.780%) / 4,077 exact (99.536%)**, compared with 4,082 / 4,074 at epoch 10.
These are deterministic-posterior-mean reconstructions, not unconditional
generation or reserved-test results. Deep prior evaluation waits for epoch 25.
Logged epoch-20 training throughput is 489.54 real molecules/second,
about 57.8 minutes for the epoch including expanded evaluation, beta 0.001,
no clipped updates and 59/256 active latent dimensions under the logged
criterion. No generation-utility or 100-epoch endpoint claim is made.

[Verified capsule003](../outputs/hpc-results/source100-first-suite-003.tar.gz),
SHA256 `7f359be1974d671bfccc253cb8fd8cfcd23f139c26f7c7d8030d30ace5029008`,
85 safe members / 84 verified files, exported 17:48:57.854140 UTC,
retains the [next training receipt](../outputs/hpc-results/source100-first-suite-003/runs/source098-large-triangle-evaluator-recovery/receipts/segment-002-receipt.json)
and [next callback receipt](../outputs/hpc-results/source100-first-suite-003/submissions/source100-triangle-auto-continuation-attempt001/receipts/after-segment-002-receipt.json).
The subsequent [17:52 UTC scheduler read](../outputs/hpc-results/source100-first-suite-003-status-detail.json)
shows segment 002 and both independent diagnostics PEND under the four-host
job-slot limit, with eligible dependencies. Old scheduler records aging out
are superseded by durable completed artifacts, not interpreted as failures.

**Historical training snapshot, 12 October 2026 09:27 DTU-local:** source098
readiness **29693747 completed** with a passed full benchmark stage and
original-reference comparison. Launcher 29693846 submitted **training
29694051 RUN**, segment `001`, batch 256 on A100 80 GB, **11 completed epochs
of 100**. The [training record](../outputs/hpc-results/source098-first-suite-003/runs/source098-large-triangle-evaluator-recovery/segments/001/20261011T203219Z_vae-train_645e1074/run.json)
uses unchanged source098/data identities and configuration SHA256
`538eff49a8bed9326ef6be4fbcdb028be4e2334bc5a32c85a00e422a2fbe0610`.
[Capsule003](../outputs/hpc-results/source098-first-suite-003.tar.gz), SHA256
`e6bf491eb53b4f326f777a10317762fd51de826c715c6e8ade97124d38c31b7e`,
has 71 safe members / 70 verified files and retains passed
[readiness](../outputs/hpc-results/source098-first-suite-003/runs/source098-large-triangle-evaluator-recovery/stage-completions/benchmark.json),
[reference comparison](../outputs/hpc-results/source098-first-suite-003/runs/source098-large-triangle-evaluator-recovery/reference-validation/comparison.json)
and [actual training receipt](../outputs/hpc-results/source098-first-suite-003/runs/source098-large-triangle-evaluator-recovery/receipts/segment-001-receipt.json).

The [epoch metrics](../outputs/hpc-results/source098-first-suite-003/runs/source098-large-triangle-evaluator-recovery/segments/001/20261011T203219Z_vae-train_645e1074/metrics.jsonl)
show reconstruction validation loss 0.00399230 at epoch 1 and 0.00040907
at epoch 11. The epoch-10 expanded reconstruction panel has **4,082/4,096
valid (99.658%) / 4,074/4,096 exact (99.463%)**. Routine epoch-11 chemistry
checks cover only 256 molecules, valid/exact 256/256. None is the reserved
final test or unconditional-prior generation; the latter awaits the deep
epoch-25 panel. Observed full epochs take 3,443.96–3,484.37 seconds,
training-only 485.37–491.03 real molecules/second; each has 6,596 updates.
One update clipped at epoch 1, none at 2–11. Current-model generation
utility and the 100-epoch endpoint are not established.

**Automatic continuation authorized, 12 October 2026:** continue the same
source098 scientific run serially toward 100 epochs while checks pass,
then perform the reserved final evaluation. The external metadata controller
must authenticate completed readiness and segment health/completion,
unchanged source/data/configuration, checkpoint/RNG/optimizer continuation
and quota before each submission, and stop on failed checks. This extends
the first-segment-only launch scope below; no scientific settings or failed
guards are changed. Source100's external controller is **armed**:
**CPU 29699487 DONE**, 71 metadata checks passed in 0.62 seconds;
the first callback 29699514 authenticated segment 001 and queued **training
29703164 / callback 29703165**, both PEND. Verified first-segment completion
and the scheduling handoff are recorded above. Source099's separate observer failed
metadata capture, not a causal model test, as retained in [docs51](51_defog_matched_pilot.md).

## Automatic serial continuation guide, 12 October 2026

Source100 is an immutable **external metadata operator**, using the pinned
existing source098 launcher and unchanged scientific runtime. Its
[archive](../outputs/source-releases/source-100-triangle-auto-continuation-attempt001.tar.gz)
SHA256 is `ee8fe26a15280982f21a1fb966b6738e34ac30126a44771014a54fc820b4909e`;
operator manifest `9cf9aa3e01b1a1d8c2c9ef2dc75976a540c03728f6d3c48287f7ecd558aa57b4`;
[policy](../outputs/hpc-submissions/source100-policy.json)
`a689eb75134fe73a52cda3fc220e0480797fec9d5bdde562f41ebe007485d77d`.
The eight packaged members are controller/health modules, CPU metadata
fixtures and an operator manifest. Release/control paths are
`/work3/s234818/latentmol/releases/source-100-triangle-auto-continuation-attempt001`
and `/work3/s234818/latentmol/submissions/source100-triangle-auto-continuation-attempt001`.
CPU 29699487 passed 71 metadata checks in 0.62 seconds, with final static
review of the sealed bytes. Arming queued **29699514**, one CPU / 4 GB /
15 minutes, `ended(29694051)`. [Capsule002](../outputs/hpc-results/source100-first-suite-002.tar.gz),
SHA256 `6e413205daf07d6cb825b7989000667bcaf2f795d0578e1973213290992fe8c2`,
71 safe members / 70 verified files at 09:53:52.796255 DTU-local, retains the
[CPU pass](../outputs/hpc-results/source100-first-suite-002/submissions/source100-triangle-auto-continuation-attempt001/cpu-passed.json)
and [callback receipt](../outputs/hpc-results/source100-first-suite-002/submissions/source100-triangle-auto-continuation-attempt001/receipts/after-segment-001-receipt.json).
At that 09:53 snapshot, training was RUN at 11 epochs and the first handoff
was pending. The current capsule003 now verifies segment 001 completed at
epoch 20 and the next pair submitted; segment 002 and final outcomes remain
pending.

1. A CPU callback waits on `ended(current_training_job)` so it can inspect
   either success or failure. Missing scheduler history is not success.
2. It authenticates the durable allocated completion, artifact hashes and
   contiguous full epoch telemetry. It checks unchanged source/data/config/
   runtime and checkpoint-parent identities, the fixed 100-epoch objective,
   finite values, expected milestone diagnostics and available quota.
   Quota must cover 2 GiB + 64 MiB per remaining epoch + 256 MiB per possible
   remaining allocation. At the first handoff (15:44 UTC), official usage
   was 87.78 / 100 GiB, leaving 12.22 GiB against 11.75 GiB required;
   the check passed and must run afresh each time. This is headroom checking,
   not a disk reservation.
3. If checks pass below epoch 100, it submits one next GPU segment using
   the existing trainer, plus one CPU callback for that new segment.
   Optimizer and dedicated RNG streams are restored and verified by the
   unchanged scientific trainer. Allocations remain serial, capped at 20
   segments; no scientific setting is rewritten by the operator.
4. At the 100-epoch endpoint it submits explicit `final-auto100` held-out
   evaluation. The reserved test is not used to choose interim checkpoints.

Any failed or unknown outcome stops continuation and retains evidence;
there is no automatic blind retry. Health checks do **not** require
monotonically improving quality: finite but noisier validation values remain
reported data. CPU fixture checks cover metadata validation and scheduling
behavior only, not new model numerics. Actual arming, callbacks, segments
and final outcomes must have distinct ledger receipts.

**Historical recovery submission, 11 October 2026:** source098 evaluator/cache
recovery is **sealed/uploaded/hash-authenticated**, with **CPU 29693717 DONE**,
102 passed / six CUDA-only skips in 47.80 seconds, and allocated **cache
validation 29693724 DONE**, exact counts and old manifest preserved. Fresh **GPU
benchmark/reference 29693747 PEND** was submitted only after the durable
cache pass. Its [binding](../outputs/hpc-submissions/source098-large-triangle-bindings.json),
SHA256 `14ae640aa68a9c776c68c8c17f0343ffc8096c7dd394687d0b5df2acdc7f9949`,
pins five source095 prepared-cache proofs and preserves the same data/original
reference. Old worker-only passes do not satisfy the fresh GPU readiness
stage. **Metadata-only controller 29693846 PEND**, dependent on
`done(29693747)`, represents the conditional launch authorization. It must
authenticate the completed CPU/cache/full benchmark and original reference,
unchanged archive/manifest/binding, fresh official quota at least 10 GiB and
a unique execution intent before calling the existing submitter for segment
`001`, run `001`. No scientific training GPU job or epoch exists yet.
Later 23-hour serial segments still need inspected continuation.
Independent source099 isolated-head
diagnostic CPU passed 81 engineering checks; its native result is pending,
as recorded in [docs51](51_defog_matched_pilot.md). Verified
[capsule001](../outputs/hpc-results/source098-first-suite-001.tar.gz), SHA256
`0246936e20fda2b8986c5a929759cabafdee11b0a7e45fc1f032266c900eda49`,
contains 29 safe members / 28 verified files, with passed current-source
[CPU](../outputs/hpc-results/source098-first-suite-001/runs/source098-large-triangle-evaluator-recovery/stage-completions/cpu.json)
and [cache](../outputs/hpc-results/source098-first-suite-001/runs/source098-large-triangle-evaluator-recovery/stage-completions/cache.json)
stage-completion receipts. Latest [capsule002](../outputs/hpc-results/source098-first-suite-002.tar.gz),
SHA256 `7fe378d8e9e667e93f6179b1fc453567c56b50ea376a3331504f58a3f4084b84`,
has 36 safe members / 35 verified files, exported 19:55:48.534072 UTC.
It retains the [gated-launch receipt](../outputs/hpc-results/source098-first-suite-002/submissions/source098-gated-training-launch/receipts/gated-launch-receipt.json)
and [queue-time quota](../outputs/hpc-results/source098-first-suite-002/submissions/source098-gated-training-launch/quota-queue.json):
86.81 / 100 GiB used, 13.19 GiB free at 19:54:44 UTC, to be checked again
at launch. Readiness GPU 29693747 and source099 native 29693807 remain PEND.
The [operator](../outputs/hpc-submissions/source098-gated-training-launch.py)
and [static receipt](../outputs/implementation-checks/source098-gated-launch-static001.json)
are metadata-only; scheduler DONE never substitutes for readiness artifacts.

**New authorization, 11 October 2026:** the user authorized the reference
evaluator correction/replay and the joint 100-epoch training launch once
readiness passes. The isolated affinity-head diagnostic is a separate
authorized investigation. Source098's identity/submission is recorded above;
no successful replay or training submission is claimed here.

The [source098 archive](../outputs/source-releases/source-098-triangle-evaluator-recovery-attempt001.tar.gz)
has SHA256 `08f55c8b81f82f3952caae06f57735ccd03785922832ce3c2242c425e3cde792`;
manifest `e8fe9b66df2224672c1c0ef69ad71c29aefab1ebfa7e353ea28536fc746bc405`.
Its 477-member snapshot contains source095 plus six evaluator/evidence/stage
helper files, excluding the draft diagnostic. Remote release is
`/work3/s234818/latentmol/releases/source-098-triangle-evaluator-recovery-attempt001`;
campaign is `/work3/s234818/latentmol/runs/source098-large-triangle-evaluator-recovery`.
The [static receipt](../outputs/implementation-checks/source098-evaluator-recovery-static001.json)
records 268 AST parses / two JSON files under local Python 3.12.14, with no
numerical/project imports. Static checks are not allocated readiness validation.

**Current result, 11 October 2026 21:13 DTU-local:** source095 preparation
**29683909 passed**, return code 0, with **1,688,539 train / 20,000 validation /
20,000 test**, exactly. Manifest SHA256 is
`f246cc93805b73a557cd0c09a78891ac8410c465192db615ce0b6709218c8c0e`.
GPU batch workers passed and selected **256**; batch 512's throughput gain
was below the declared 10% threshold. Actual model counts match 3,941,860 total.
The enclosing benchmark/reference wrapper **29683914 failed**, return code 1,
when reference-prior summarization tried to fingerprint a `None` molecule
after `MolFromSmiles` could not parse an output. Full reference evaluation and
allocated benchmark-stage completion are absent. An evaluator fix/replay
is needed before training; **no 100-epoch training is submitted**.

The user authorized full dataset preparation and GPU batch-size benchmarking,
and subsequently the conditional training launch described above.
The exact counts match all 1,728,539 supported inventory identities.
The [authenticated binding](../outputs/hpc-submissions/source095-large-triangle-bindings.json),
SHA256 `8231878eb667e834d53b751c9007ae2185d0d4e3518724f4f96de196a297ecb2`,
pins source094's report, allocated completion, intent, receipt and binding,
with unchanged preprocessing-source checks required before preparation.
Source095 is **sealed/uploaded/hash-authenticated; CPU 29683906 DONE**:
**90 passed / six CUDA-only skips in 40.88 seconds**. Preparation scheduling
and runtime same-source receipt/manifest checks remain authenticated.
No job was added, changed or canceled by this read-only status followup.
Full training awaits successful reference/stage replay. The original checkpoint and
protected split identities remain preserved; reconstruction plus VAE KL has
no chemistry loss or extra prior training points. See the gate details below.

[Historical preparation capsule003](../outputs/hpc-results/source095-first-suite-003.tar.gz),
SHA256 `080138bc17680e840f46ef41e87598b8d8030fb9bcd4b05091c38d0a34555105`,
contains 27 safe members / 26 verified files, exported
`2026-10-11T10:01:30.910097+00:00`. At that snapshot preparation was RUN and
benchmark PEND; actual prepared counts and GPU outcomes were pending. The separate
source096 observer completed zero of two cases due to an unequal-ensemble-
depth instrumentation assumption, retained in [docs51](51_defog_matched_pilot.md).
It does not change this dataset/benchmark release or establish a Boltz remedy.

## Prepared cache and bounded batch measurements, 11 October 2026 21:13 DTU-local

[Capsule004](../outputs/hpc-results/source095-first-suite-004.tar.gz), SHA256
`26d0c783defb233a0b3c79a6c7758b13614bd6998098e827b945a4000dae3fef`,
contains 46 safe members / 45 verified files, exported
`2026-10-11T19:13:00.440050+00:00`. The
[prepare run](../outputs/hpc-results/source095-first-suite-004/runs/source095-large-triangle-prepare-benchmark/job-records/20261011T083317Z_prepare_82620e1a/run.json)
and [allocated completion](../outputs/hpc-results/source095-first-suite-004/runs/source095-large-triangle-prepare-benchmark/stage-completions/prepare.json)
establish return code 0, 08:33:17.951726–10:24:24.682669 UTC and strict exact
split-count/manifest authentication. The prepared manifest SHA256 is
`f246cc93805b73a557cd0c09a78891ac8410c465192db615ce0b6709218c8c0e`.

All four [worker reports](../outputs/hpc-results/source095-first-suite-004/runs/source095-large-triangle-prepare-benchmark/benchmark/passed.json)
passed, with finite updates, zero clipped updates and changes to both encoder
and decoder weights. Each measured 20 updates after five warmups; weights
were discarded. These are short warmed timings, not complete training epochs.

| Real batch | Mean seconds/update, repeats 1 / 2 | Molecules/second, repeats 1 / 2 | Peak allocated bytes |
|---:|---:|---:|---:|
| 256 | 0.56690 / 0.57285 | 451.58 / 446.89 | 34,233,211,904 |
| 512 | 1.09055 / 1.08781 | 469.49 / 470.67 | 68,344,875,008 |

The predeclared throughput rule selected 256 because 512 gained less than
10%. The [selected config](../outputs/hpc-results/source095-first-suite-004/runs/source095-large-triangle-prepare-benchmark/benchmark/selected-training.yaml)
has SHA256 `49c4747bbeff7ed4a682dede0632acd9a19e9659c039dde60533bd821e81f1fd`.
Its conservative 7,378.53-second epoch forecast is explicitly extrapolated
from these short warmed batches, not an epoch observation.

For timing context, the untouched original batch-64 joint VAE completed
20 continuation epochs of 215,000 real examples in 7,163.04 seconds on an
A10: **600.30 real presentations/second including phase overhead**. The
later frozen-encoder dense chemistry decoder's six full warmed L40S updates
were **157.91 / 165.18 real molecules/second**, also processing 64 fresh
prior points per update. These differ in architecture, hardware, objective
and timing scope from this A100 benchmark. The [ledger](05_experiment_ledger.md)
links the exact retained records and denominators; these are not a matched
hardware or architecture speed comparison.

The [benchmark wrapper](../outputs/hpc-results/source095-first-suite-004/runs/source095-large-triangle-prepare-benchmark/job-records/20261011T102800Z_benchmark_e1280f6f/run.json)
failed return code 1 at 10:34:53 UTC. Its
[traceback](../outputs/hpc-results/source095-first-suite-004/runs/source095-large-triangle-prepare-benchmark/job-records/20261011T102800Z_benchmark_e1280f6f/command.log)
shows `large_vae_evaluation.summarize` calling `GetFingerprint(None)` after
an output failed `MolFromSmiles`. Thus the worker `benchmark/passed.json`
does not establish completed reference validation or the enclosing stage;
no benchmark stage-completion receipt is present. Evaluator fix and replay
are the next required steps, not a passed recovery or submitted training.
Source095 source/data/original-reference identities and all prior failures
remain preserved. Local retrieval verification used Python 3.12.14 / Windows
11 byte/hash/JSON reads only; numerical preprocessing/benchmarking ran on HPC.

The [source095 archive](../outputs/source-releases/source-095-triangle-prepare-benchmark-attempt001.tar.gz)
has SHA256 `794bfa225cb0286484a1ecc3fb37dfb88a99b391b24f17c56aa56aa8f7578d69`;
scientific manifest `f1017a9d028bb29a4882185fa47b0dc3fe92b6da92aaae971f30e888e4581808`.
HPC staging authenticated 476 members. Its
[packaging receipt](../outputs/source-releases/source-095-triangle-prepare-benchmark-attempt001.tar.gz.json)
records the source094 base plus six named finalized preparation/helper/config/
fixture files, including their uncommitted changes. Unfinished source096
diagnostic work was excluded. The [snapshot builder](../outputs/hpc-submissions/build-source095-snapshot.py)
and [static receipt](../outputs/implementation-checks/source095-prepare-benchmark-static001.json)
retain metadata-only packaging and 267 Python AST / two JSON parses under
Python 3.12.14 / Windows 11; no scientific imports or numerical tests locally.
The first upload was rejected before transfer with `Permission denied (password)`.
After the user reported restored access, retry, authenticated staging and CPU
submission succeeded, without VPN/SSH setting changes. That failed attempt
remains history and supplies no scientific result.

[CPU capsule001](../outputs/hpc-results/source095-first-suite-001.tar.gz),
SHA256 `9bf953f2b6dbc5d77974739d2e1f5ebb4df4057413e33c341debd8e1ccf56cd7`,
has 17 safe members / 16 verified files, exported
`2026-10-11T08:32:46.372819+00:00`. The
[completed run](../outputs/hpc-results/source095-first-suite-001/runs/source095-large-triangle-prepare-benchmark/job-records/20261011T083146Z_cpu_b4c7deb6/run.json),
[command log](../outputs/hpc-results/source095-first-suite-001/runs/source095-large-triangle-prepare-benchmark/job-records/20261011T083146Z_cpu_b4c7deb6/command.log)
and [allocated completion](../outputs/hpc-results/source095-first-suite-001/runs/source095-large-triangle-prepare-benchmark/stage-completions/cpu.json)
authenticate the fresh CPU result. The six CUDA-only skips are expected on
the CPU allocation; this is not a GPU model/batch result. Local evidence
verification was byte/hash/metadata only. Subsequent stage states/receipts
will be appended when exported; no 100-epoch training was submitted.

**Latest stage snapshot, 11 October 2026 10:36 DTU-local:**
[capsule002](../outputs/hpc-results/source095-first-suite-002.tar.gz), SHA256
`2cb3922b18985fdfc44706602e887d25a5096f74933ce6cceea9c6911beafbaf`,
contains 27 safe members / 26 verified files, exported
`2026-10-11T08:36:03.079451+00:00`. Its
[scheduler snapshot](../outputs/hpc-results/source095-first-suite-002/scheduler_snapshot.json)
confirms CPU DONE / preparation RUN / benchmark PEND. The
[prepare receipt](../outputs/hpc-results/source095-first-suite-002/runs/source095-large-triangle-prepare-benchmark/receipts/prepare-receipt.json),
[benchmark receipt](../outputs/hpc-results/source095-first-suite-002/runs/source095-large-triangle-prepare-benchmark/receipts/benchmark-receipt.json)
and [running preparation record](../outputs/hpc-results/source095-first-suite-002/runs/source095-large-triangle-prepare-benchmark/job-records/20261011T083317Z_prepare_82620e1a/run.json)
preserve actual identities and commands. Preparation started at 08:33:17 UTC;
no error is reported at this snapshot, but prepared counts/completion remain
pending. Capsule001's CPU proof remains retained; neither accepted jobs nor
inventory authentication establish a completed preparation or GPU benchmark.

**Completed first-suite snapshot, 11 October 2026:** the architecture, memory-mapped data preparation,
benchmark/training/evaluation runner and serial submission interfaces are
implemented locally and **not certified for training**. The user subsequently
reported restored HPC access and authorized the runs needed for this first
suite, superseding the preceding implementation-turn outage restriction.
Source094 fixture-only recovery is sealed/uploaded/hash-authenticated:
triangle CPU **29680599 DONE**, **50 passed / six expected CUDA-only skips
in 38.79 seconds**, with an authenticated allocated completion receipt.
Inventory **29680603 DONE**: **1,936,962 raw rows / 1,728,539 supported
unique identities / 208,423 representation-roundtrip rejections / zero
duplicates**, without graph arrays or new split assignment. Source093 triangle
CPU **29680547 EXIT 1** remains retained: 49 checks passed, six skipped and
one continuation fixture failed because it passed a directory as the protocol
file. Production correctly hashes that file; source094 changes only the
fixture, with production models/data/runner unchanged. Replay CPU **29680549 DONE**:
42 checks passed in 8.84 seconds. Exact-state GPU replay **29680551 DONE**,
both observers completed with frozen/native parity, but the original failed
guards remain uncertified and the Boltz gradient discrepancy is unresolved.
At that snapshot, full preparation, benchmark and training were not queued;
the source095 readiness entry above now records the feasible requested counts.
Source092 remains a local historical package, not an uploaded/submitted job.
The
[model config](../configs/vae_large_triangle.yaml),
[protocol](../configs/large_triangle_protocol.json) and
[bindings template](../configs/large_triangle_bindings.template.json) exist;
the template is distinct from the authenticated source094 binding. Static review
and immutable packaging prepare a later allocated run, not a passed experiment.

## Scope and historical references

Train the new graph VAE's **encoder and decoder jointly** on supported real
molecules from the approximately 1.9-million curated corpus, through a
declared **100-epoch maximum budget**. This is a scoped user-authorized
extension beyond the historical frozen-encoder decoder studies. The training
objective is real-molecule reconstruction plus VAE KL, with posterior latent
sampling. It has **no additional unconditional random-prior training points,
chemistry proxy, repair-target loss, DeFoG teacher or element-distribution
anchor**. Random-prior generation is an evaluation, not another training arm.
The implemented reconstruction config sets `size_weight=0`, unlike the
historical size-MSE weight 0.1; the size diagnostic remains recorded. Thus the
historical comparison also changes this loss setting.

Preserve the untouched epoch-89 VAE and its data/checkpoint identity. The
historical chemical decoder may be a separately labeled evaluation reference;
it supplies neither weights nor targets to this new objective. Different
architecture, corpus, joint-training regime and compute budget confound a
comparison against those historical checkpoints. It is not a causal
triangle-only architecture ablation. Any newly trained decoder must be frozen
before later molecular latent optimization; external scorer boundaries in
[docs23](23_frozen_model_plugin_contract.md) remain unchanged.

## Final architecture and expected counts

The implemented `ModelConfig` variant is `triangle_refine`, in
[triangle.py](../src/latentmol/models/triangle.py). It uses 96-wide refinement
MLPs, superseding the earlier 192-wide discussion. The counts below were
analytical expectations and are now **confirmed by allocated GPU benchmark
workers**; they are not trained-model quality results.

| Component | Declared structure | Confirmed parameters |
|---|---|---:|
| Encoder | Four bond-aware 96-wide message blocks; flattened 50×96 state →320; two separately parameterized single-linear 320→320 residual updates with affine pre-LayerNorm; μ/log-variance heads of width 256 | 2,173,216 |
| Decoder without triangles | Biased 256→4,800 slot projection; initial 96→96→96 slot residual; pair initialization 288→48→48; four unshared refinement rounds and categorical heads | 1,735,044 |
| Triangle additions | Four separate 24-wide blocks, 8,400 parameters each | 33,600 |
| Final triangle decoder | Decoder plus all four triangle blocks | 1,768,644 |
| Final VAE | Encoder plus final triangle decoder | **3,941,860** |

Each refinement round has pair MLP `336→96→96→48` and node MLP
`96→96→96→96`, each with two SiLU activations. Four-head pair-aware
attention retains 24 features per head. Stabilizing nonaffine post-residual
normalizations do not add trainable parameters. The two encoder residual
updates each have one 320→320 linear; they are not two two-layer MLPs.
The decoder has **no global 320→512→320 trunk or global bond-context slot**:
the direct latent-to-slot projection replaces that historical trunk.

The model retains 50 canonical heavy-atom slots. Hidden nodes have shape
`[B,50,96]`, hidden pairs `[B,50,50,48]`, and the triangle
contraction passes information between pair `(i,k)` and pair `(k,j)` through
slot `k`, with declared symmetry/occupancy handling. It must not permanently
materialize a three-atom hidden tensor. Updated pair states feed both
pair-aware node attention and the final bond head; they are not categorical
bond probabilities. Final logits have atom shape `[B,50,10]`, charge
`[B,50,5]` and bond `[B,50,50,5]`. The final bond head takes the 48-wide pair
state plus 288 symmetric endpoint features, then `336→96→5`. Atom and
charge heads remain `96→10` and `192→5`, with soft atom/charge feedback.
The shared graph boundary enforces undirected bonds, occupied endpoints and
no self-bonds; hard conversion stays outside autograd.

For comparison, the authenticated original epoch-89 model has 1,901,600
encoder / 2,201,628 decoder / 4,103,228 total parameters. Its
[training record](../outputs/hpc-results/source078-dense100/extracted-single003/original/training-run.json)
pins `stable_deep_pair_refine`, two rounds and its historical global trunk.
The optional historical global-bond variants are not this reference. Neither
parameter count alone nor triangle information flow guarantees validity.

## Corpus preparation and split protection

There is **no raw acquisition/download stage** in this pipeline. Its
[large_data v2 CLI](../src/latentmol/large_data.py) consumes a supplied existing
SMILES/CSV file, uses SQLite canonical-identity deduplication and writes
memory-mapped label arrays. All chemistry preprocessing and cache construction
are later HPC work. Pin raw source/version/digests, preparation source and config,
canonical chemical-identity policy, graph-schema identity and split seed.
Deduplicate the curated corpus by the declared canonical chemical identity
and retain present protected original train/validation/test identities in
their original roles; **original validation/test identities cannot enter
new training, and original training identities cannot enter new holdouts**.
Record within-corpus duplicates, protected
overlaps and every representation rejection separately. Filtering means the
supported training count may be below the nominal 1.9 million.

Public MOSES source split labels are ignored: this is a declared **custom
canonical-identity resplit**, preserving the protected historical roles,
not the official MOSES train/test benchmark. The implemented `--mode inventory`
is a separate allocated stage before full preparation. It counts supported
canonical identities and protected overlaps, records rejections and identities,
and does not allocate graph arrays or assign new split roles. Its receipt
must guide a feasible new sealed split protocol; inventory is not a training
cache, benchmark or completed preprocessing experiment until it actually runs.

**Raw acquisition update, 11 October 2026:** the execution owner separately
downloaded/authenticated MOSES at upstream commit
`dd7ed6ab38e23afd3ef5371d67939a1760bd8599` to
`/work3/s234818/latentmol/raw/moses-v1-dd7ed6ab38e2.csv`, column `SMILES`,
84,482,588 bytes, SHA256
`bb47a94d347afd476d3828b5e26dceeabc42a2d8cf92a791d00349f22fea0d8b`.
Acquisition is not allocated support inventory or a prepared dataset. The
ledger retains the pre-download quota and pending source093 gate scope;
no triangle training job is established yet.

The source095 protocol requests **1,688,539 train + 20,000 validation +
20,000 test**, exactly, based on the authenticated supported inventory.
The historical 1,900,000 + 40,000 template was infeasible even before
filtering and remains retained in the earlier source releases. Preparation
fails closed on shortage and never silently lowers training or holdout
quotas. Requested counts do not establish an assigned or prepared split:
training requires the authenticated manifest's actual counts to match the
new sealed protocol exactly.

The source095 binding selects cache
`/work3/s234818/latentmol/data/source095-moses-triangle-v2` and campaign
`/work3/s234818/latentmol/runs/source095-large-triangle-prepare-benchmark`.
It authenticates the source094 inventory report, allocated completion,
submission receipt/intent and binding, and requires the same four
preprocessing-source file hashes and exact supported-count total.
The protected original cache and untouched epoch-89 reference remain pinned.
Fresh quota is 82.49 / 100 GiB used, with 17.51 GiB free; this is a capacity
observation, not a prepared-cache result.

The shared representation is still 50 heavy-atom slots, `NO_ATOM` plus
elements C/N/O/F/P/S/Cl/Br/I, formal charges −2/−1/0/+1/+2 and
NONE/SINGLE/DOUBLE/TRIPLE/AROMATIC bond labels. Atom order, cleaning,
stereochemistry treatment, fragments and unsupported modes are declared by
the manifest. The implemented preparation removes stereochemistry before
identity deduplication and requires connected sanitized graphs with exact
canonical-SMILES round trips. It does not strip salts or normalize charges
or tautomers; rejected features/counts remain explicit. Existing limitations
are not waived by a larger corpus.
No chemistry conversion belongs inside the differentiable model path.

Host memory and disk checks must cover raw/filtered data, cache construction,
loader buffers, checkpoints and logs independently of GPU VRAM. A100 80 GB
does not certify host-memory capacity. The previously discussed 64 GB host
request and existing L40S batch timings do not prove that this corpus/new
model fits. SQLite/memory-mapped construction and loading are implemented,
but their fidelity and peak-memory behavior have not been validated on HPC.

## Training and evaluation stages

1. **Immutable preparation:** static review, manifest and source archive
   including uncommitted scientific files; later upload/hash authentication
   only when HPC access and execution authorization permit it.
2. **Allocated CPU contracts:** config/data/schema rejection fixtures,
   identity and split protections, shape/symmetry/count contracts and
   checkpoint metadata. A CPU pass cannot establish native GPU training.
3. **Allocated inventory:** after the triangle CPU gate, scan the entire
   authenticated raw source for supported deduplicated counts and protected
   memberships. Preserve counts/rejections without allocating label arrays
   or assigning new roles. Choose feasible exact counts in a new sealed
   protocol before attempting full preparation.
4. **Allocated full preparation:** with the feasible authenticated protocol,
   materialize immutable memory-mapped arrays and exact protected custom
   splits. This is a later stage; the unattainable current template is not
   queued for full preparation.
5. **Allocated native smoke, batch benchmark and reference validation:** validate finite forward,
   backward and parameter updates for both encoder and decoder, triangle
   paths and checkpoint handling. Benchmark real batches **256 and 512**
   with FP32 and measured host/GPU peaks, examples per second and both
   encoder/decoder gradients and actual parameter changes. Each batch uses
   **two repeats, five warmup and 20 measured true training updates** in
   isolated workers; benchmark weights are discarded. Require declared
   80-GB hardware and peak-memory fraction at most 85%. Choose 512 only if
   eligible and at least **1.1×** the measured throughput of 256; otherwise
   choose eligible 256. An ineligible baseline aborts rather than inventing
   an unbenchmarked batch. Earlier batch-128
   discussion is not the current benchmark prescription. Choose the training
   batch from measured throughput/quality, not the largest fitting batch.
   The three-hour benchmark allocation also runs GPU unit checks and the
   `reference-validation` stage: evaluate the authenticated original and
   optional historical chemical references once on the fixed 4,096-point
   validation panel, including deep prior/neighborhood diagnostics. These
   retained reference results support interim comparisons at epochs 25/50/75
   without opening the reserved test split.
6. **Full joint training:** fresh seed-17 initialization, real reconstruction
   plus VAE KL, FP32 and loader workers 0. The config declares initial learning
   rate 0.001, weight decay 0.00001, clipping 5, β 0.001 with 20-epoch warmup,
   later learning-rate milestones, padding weight 0.2, no-bond weight 0.1,
   charge weight 1, bond weight 2 and size-loss weight 0. The benchmark writes
   the sealed selected-batch config; no benchmark weights become initialization.
   All numerical work remains on DTU HPC.
7. **Scheduled assessment:** **every epoch** validates 4,096 held-out
   reconstructions, selects validation reconstruction/validity checkpoints
   and evaluates a fixed 256-point chemistry probe. An additional expanded
   panel runs every **10 epochs**, with deeper
   predeclared analysis at **25/50/75/100**. These
   milestones do not authorize repeated use of the sealed final test to
   tune architecture, loss, batch or checkpoint selection.
8. **Reserved final evaluation:** explicit `final` stage only after epoch 100,
   evaluating the 20,000-point test split for epoch-100, best validation
   reconstruction and best validation validity checkpoints, plus authenticated
   original and optional historical chemical references. Each shares the
   same evaluation; the final test is not available to monitoring or selection.
   Each reference requires both `--reference label=path` and
   `--reference-sha256 label=hash`; candidate names are protected from
   reference-label collisions.

Deep evaluation includes a 4,096-point prior panel, argmax plus four samples,
sampled reconstruction, perturbations of 128 declared latent points and 32
interpolation pairs, composition with invalid/empty/duplicate outputs retained,
and valid-molecule fingerprint similarity to declared anchors. Slot-aligned
edit counts are **not minimum graph-edit distances**. Raw graphs are saved
as JSONL gzip with sparse edges. There is no graded/search chemistry training
penalty and no raw-invalid molecule is replaced by a repaired evaluation.

The final comparison should report reconstruction component losses, whole
graph exactness/validity, occupancy/element/charge/aromatic composition,
unfiltered argmax and sampled validity, uniqueness/novelty, rejected modes
and physical costs. Repair witnesses, if audited, remain upper bounds in a
declared bounded search; they do not replace raw output validity. Preserve
failed, stalled, censored or unattempted stages and every scheduled endpoint.
Final test data remain reserved until the predeclared evaluation decision.

## Execution history and limits

The implemented [large_vae CLI](../src/latentmol/large_vae.py) has stages
`benchmark`, `benchmark-worker`, `segment`, `recover`,
`reference-validation` and `final`, using the
shared training observer and recorded continuation RNG state. The
[submission helper](../scripts/hpc/submit_large_triangle.py) has stages
`cpu`, `inventory`, `prepare`, `benchmark`, `segment`, `recover` and `final`, dispatched
by [large_triangle_stage.py](../scripts/hpc/large_triangle_stage.py). Its
interface accepts `--release`, `--archive-sha256`, `--bindings`,
`--bindings-sha256`, `--receipts`, `--run-id`, `--segment-id`, `--after-stage`
and `--submit`. It defaults to a verified dry-run plan; incomplete bindings
are not launch-ready. The source095 helper adds opt-in
`--queue-after-prepare` for the benchmark stage only. Submission still
requires a completed fresh CPU gate; it binds the same-release preparation
receipt, its SHA256 and job identity and schedules `done(prepare_job)`.
On the allocated GPU node, preparation completion and artifact hashes are
independently authenticated **before GPU tests, model construction or data
loading**. Queuing a benchmark is not a preparation or benchmark pass.
The source093/source094 first-suite receipts below remain historical gate
evidence; source095's changed helper still needs its own allocated CPU checks.

The local stdlib-only checker
`scripts/check_large_triangle_syntax.py --output outputs/implementation-checks/source092-large-triangle-static001.json`
passed AST parsing of 265 Python files and JSON parsing of the two new configs.
Its receipt records file hashes, Python 3.12.14/Windows and exact command. It
imports no scientific module; numerical fixtures remain unexecuted. The local
archive `outputs/source-releases/source-092-large-triangle-vae-attempt001.tar.gz`
and its adjacent receipt are prepared for later upload, not cluster validation.

Normally one serial stage is submitted at a time; the explicit benchmark
queue option waits for authenticated preparation as described above.
GPU stages request A100 80 GB,
four cores and 64 GB host RAM. Training segments use a **22-hour controller
budget inside a 23-hour allocation**, checkpointing at epoch boundaries and
continuing with persistent optimizer/RNG/checkpoint identity. A failed monitor
requires explicit `recover`; it is not silently skipped or treated as a
completed segment. Durable allocated stage-completion receipts protect gates
from scheduler history expiry by authenticating actual artifact hashes,
job/source identity and completed allocated execution. A missing scheduler
status alone is not success. The ledger preserves local source/archive
identities; raw/reference and inventory bindings are pinned for source095,
while prepared-cache outcomes and later allocation outcomes remain pending. Every
submitted/completed/failed stage belongs in the [ledger](05_experiment_ledger.md).
Packaging or static syntax checks are not numerical validation. Existing
DeFoG/Boltz jobs and historical decoder studies are not mutated by this plan.
No binding, synthesis, selectivity or validity guarantee follows from this
architecture or its larger training corpus.

## Source093 first-suite submission, 11 October 2026

The [immutable source093 archive](../outputs/source-releases/source-093-first-suite-attempt001.tar.gz)
SHA256 is `78405ec51fb571c8e9f0ad3fc41a98a0cc1b1fbba16e67b1724239117499c0e2`,
manifest `485936effcd12fff758e5b95a578ff486cb3a3b579ea5640fdf6cd150f6679c3`.
HPC staging authenticated 474 safe members at
`/work3/s234818/latentmol/releases/source-093-first-suite-attempt001`.
[Bindings](../outputs/hpc-submissions/source093-large-triangle-bindings.json)
SHA256 `ddc07832de763d4174da0eae4a25c80bf8749ae6020520f3cd775dd948d078f2`
pin the public raw CSV, protected `zinc215k-seed17-v1` cache manifest
`0ba04cf46b7b6a94ea799fbd52596ed60c2a82493a49a56e4d9d67f284f200ea`,
and original epoch-89 checkpoint
`1893ddc183473fd4f27c6cd921d8b302f67ab388b660ada44bf3a2d6d5e28088`.
Triangle campaign is `/work3/s234818/latentmol/runs/source093-large-triangle-first-suite`;
triangle CPU **29680547** uses four cores, 8 GB per core and one hour.
Inventory is not submitted until that gate passes; its allocated limit is
12 hours, four cores and 16 GB per core. Full preparation/training are not
submitted.

The separate replay plan SHA256
`85b103725e2a590fe19a00a5aae704b0f3eb1ca856547426a1f913f6820b85ba`
pins the source091 control/guided step-15 failures and original source/config/
state/target evidence. Replay CPU **29680549** uses 30 minutes, two cores
and 4 GB per core; native **29680551** waits `done(29680549)` and uses one
A100 80 GB, four cores / 16 GB per core, managed MPS and a four-hour limit.
It observes exactly two saved states with no optimization or new DeFoG calls.
A submitted diagnostic is not an approved historical guard or a completed
long pilot. CPU/native outcomes and exported receipts were pending at that
initial submission snapshot; the verified followup below supersedes it.
The [ledger](05_experiment_ledger.md) retains operational metadata failures
and exact evidence links as they become available.

## Verified first CPU outcomes and fixture recovery, 11 October 2026

[Capsule001](../outputs/hpc-results/source093-first-suite-001.tar.gz), SHA256
`5025302bacd244fee3faffe18e76396caf6d639c3b7f10a94f0e3bf43647a02b`,
contains 29 safe members / 28 verified files, exported
`2026-10-10T22:52:00.773923+00:00`. Its
[verification receipt](../outputs/hpc-results/source093-first-suite-001.tar.gz.verification.json)
records local Python 3.12.14 / Windows 11 metadata/byte checks only. Allocated
tests used Python 3.11.9 / Torch 2.5.1+cu121 / pytest 8.3.5.

The [triangle CPU run](../outputs/hpc-results/source093-first-suite-001/runs/source093-large-triangle-first-suite/job-records/20261010T225020Z_cpu_2ebf0cb4/run.json)
and [command log](../outputs/hpc-results/source093-first-suite-001/runs/source093-large-triangle-first-suite/job-records/20261010T225020Z_cpu_2ebf0cb4/command.log)
retain 1 failed / 49 passed / 6 skipped in 42.52 seconds and failed return
code 1. The sole failure is
`test_segment_persists_progress_using_bound_selected_config_identity`, whose
fixture passed `tmp_path` as `protocol_path`; the file-hash guard correctly
raised `IsADirectoryError`. The planned fix uses actual pinned config,
protocol and gate files in the fixture, with no production math/settings
change. At that source093 snapshot, source094 identities/submission were
pending; the verified source094 followup below supersedes this state.
This does not certify the triangle workflow or permit
inventory/preparation to bypass its CPU gate.

The separate [completed replay CPU run](../outputs/hpc-results/source093-first-suite-001/runs/20261010T225037Z_source-093-first-suite-attempt001-cpu_215f6761/run.json)
and [log](../outputs/hpc-results/source093-first-suite-001/runs/20261010T225037Z_source-093-first-suite-attempt001-cpu_215f6761/command.log)
retain **42 passed in 8.84 seconds**, return code 0. The
[scheduler snapshot](../outputs/hpc-results/source093-first-suite-001/scheduler_snapshot.json)
records triangle EXIT / replay CPU DONE / native replay PEND. These fixtures
establish engineering behavior only; exact-state GPU replay and a completed
long DeFoG pilot remain unproven. At that snapshot, no triangle inventory,
full preparation, batch benchmark or training was submitted.

## Historical source094 recovery submission, 11 October 2026

The [source094 archive](../outputs/source-releases/source-094-triangle-gate-recovery-attempt001.tar.gz)
SHA256 is `5eb5d348c37bc23ab820226fdd08679d5fcee1e70b308881d3f9b334347c0c20`,
manifest `86ed4bdb4c348d1f29b159ba77af9a308c3bdc94eab1d94383f986ccbfbdc9e0`.
HPC staging authenticated 474 safe members. The only scientific-file change
from source093 is `tests/test_large_vae_pipeline.py`: real config/protocol
fixture files and stronger persisted-hash assertions. Production model, data,
loss and runner are unchanged. The
[static receipt](../outputs/implementation-checks/source094-triangle-recovery-static001.json)
retains 265 AST / two JSON parses, not numerical validation.

Triangle CPU **29680599** is submitted for the same one-hour/four-core/
8-GB-per-core gate. Binding SHA256
`11fc11aa10acb1debc2264c98a7911080b419705c4e17dc37fee73d54bfdf9d1`
uses the unique campaign `/work3/s234818/latentmol/runs/source094-large-triangle-first-suite`.
Inventory awaits a completed CPU pass; no full preparation/training is
submitted. The source093 replay CPU/native jobs are retained without duplicate
submission. Actual source094 CPU outcome and exported receipts remain pending.

## Verified source094 CPU pass and queued inventory, 11 October 2026 00:55 DTU-local

[Capsule001](../outputs/hpc-results/source094-first-suite-001.tar.gz), SHA256
`0969d54647539bb2640521829e1d62468c91ca4e2172990eac17168358a8d303`,
contains 22 safe members / 21 verified files, exported
`2026-10-10T22:55:18.934930+00:00`. The
[verification receipt](../outputs/hpc-results/source094-first-suite-001.tar.gz.verification.json)
records local Python 3.12.14 / Windows 11 byte/hash/metadata verification.
The [completed CPU run](../outputs/hpc-results/source094-first-suite-001/runs/source094-large-triangle-first-suite/job-records/20261010T225342Z_cpu_a26b6f73/run.json)
and [command log](../outputs/hpc-results/source094-first-suite-001/runs/source094-large-triangle-first-suite/job-records/20261010T225342Z_cpu_a26b6f73/command.log)
retain **50 passed / six CUDA-only skips in 38.79 seconds**, return code 0.
The [allocated stage completion](../outputs/hpc-results/source094-first-suite-001/runs/source094-large-triangle-first-suite/stage-completions/cpu.json)
authenticates job 29680599, source/archive/binding and result artifacts.

Model config SHA256 is
`361221d83810c3cbd61bbf2b9cba80a25099448eb100f212876fe2e79ef295ce`;
protocol SHA256
`7559f765ecc63e4af14fedf0d3121addd3d6b805c64f9d3f609ab6ac84081003`.
They remain unchanged between source093 and source094. The
[archived comparison receipt](../outputs/implementation-checks/source094-fixture-difference001.json)
proves only the fixture file changed scientifically. Source093's failed
fixture and source091's pilot failures remain retained.

The [inventory receipt](../outputs/hpc-results/source094-first-suite-001/runs/source094-large-triangle-first-suite/receipts/inventory-receipt.json)
records **29680603 submitted**, after the verified CPU pass; the
[snapshot](../outputs/hpc-results/source094-first-suite-001/scheduler_snapshot.json)
records PEND. Resources are 12 hours, four cores × 16 GB host RAM, no GPU.
Inventory writes support/rejection/protected-membership evidence only,
without graph-array allocation or split assignment. It is not yet completed
or a prepared dataset. Fresh work3 quota is **84.49 / 100 GiB used,
15.51 GiB free**; no cleanup occurred in this turn. Full preparation,
GPU benchmark and 100-epoch training remain unsubmitted pending inventory
and a feasible new sealed count protocol. Source093 replay CPU remains DONE
with 42 passes and its native GPU remains PEND at last verification.

**Running inventory followup, 11 October 2026 00:58 DTU-local:**
[Capsule002](../outputs/hpc-results/source094-first-suite-002.tar.gz), SHA256
`5c629621eb9c4712cbe403fed663dd12b7f46c0349ff7401ef77db86d6f6d8ed`,
contains 24 safe members / 23 verified files, exported
`2026-10-10T22:58:01.987437+00:00`. The
[verification receipt](../outputs/hpc-results/source094-first-suite-002.tar.gz.verification.json)
and [scheduler snapshot](../outputs/hpc-results/source094-first-suite-002/scheduler_snapshot.json)
retain inventory **29680603 RUN**, started around 00:55 DTU-local, with
its live wrapper/log and no completion yet. Source093 native replay remains
PEND at last coordinator verification. Capsule001 and its passed CPU proof
remain retained. No support counts, prepared graph arrays, GPU benchmark or
100-epoch training result is established.

## Completed inventory and replay, 11 October 2026 10:13 DTU-local

[Inventory capsule003](../outputs/hpc-results/source094-first-suite-003.tar.gz),
SHA256 `63e55402d5e96a12992dce0268ac2ffde040748fb504fba3fe3d5fff446590ef`,
contains 26 safe members / 25 verified files, exported
`2026-10-11T08:13:10.186558+00:00`; its
[verification receipt](../outputs/hpc-results/source094-first-suite-003.tar.gz.verification.json)
records byte/hash/metadata checks only. The
[completed inventory](../outputs/hpc-results/source094-first-suite-003/runs/source094-large-triangle-first-suite/inventory/inventory.json)
has status `completed_inventory_only`, inventory identity
`f2c29115ae6c15bdaa44d74a0bc11f12da93bbc1783c31c14009e4d6db7872a7`
and supported-identity digest
`c8a2c3392789c7eb27e38805ad5b7e678c52ad03d09fef5265baf9f2c6e208a4`.
It reports 1,936,962 source rows, 1,728,539 supported unique identities,
208,423 `representation_roundtrip` rejects and zero duplicates. The rejection
category alone does not isolate an underlying chemical feature. No split
labels or graph arrays were created, and source MOSES split labels stayed
ignored. A feasible new sealed protocol is required before preparation;
the 1,900,000 training target cannot be met from this supported count.
Retaining 20,000 validation and 20,000 test identities leaves a possible
full-use training count of **1,688,539**; this is arithmetic, not an assigned
split or sealed training configuration. Supported corpus graphs have neutral
formal charges and no P or I in the inventory composition. A representation
roundtrip rejection does not imply the source molecule is chemically invalid.

The [allocated run](../outputs/hpc-results/source094-first-suite-003/runs/source094-large-triangle-first-suite/job-records/20261010T225519Z_inventory_9280055a/run.json)
and [stage-completion receipt](../outputs/hpc-results/source094-first-suite-003/runs/source094-large-triangle-first-suite/stage-completions/inventory.json)
retain successful job 29680603, finishing at 01:51:55 DTU-local on 11 October
after approximately 56.6 minutes. Peak process host RSS was 568,938,496 bytes
(about 543 MiB). The estimated all-supported cache payload is 4,571,070,538
bytes (about 4.26 GiB); this is not a materialized cache or training-memory
benchmark. Fresh work3 quota is 82.49 / 100 GiB used, 17.51 GiB free, with
no cleanup this turn. Shared-filesystem free space is not the user's quota.

[Replay capsule002](../outputs/hpc-results/source093-first-suite-002.tar.gz),
SHA256 `f9876929f75c8c68c5a47eae9cf4ca05faa1d282755fcea2842b60d23ecaaf6d`,
contains 43 safe members / 42 verified files, exported
`2026-10-11T08:13:10.201622+00:00`. Job 29680551 finished at 02:23:25
DTU-local after 11 minutes 11 seconds. Both exact-state observers returned 0
and completed without timeout. Native parity, frozen states, parent artifacts
and original random directions were preserved; repeated values/gradients
and grad-enabled/no-grad baseline differences were zero.
All paired plus/minus endpoints also have identical grad-enabled/no-grad
outputs, graph hashes and contact selections (18 control / 36 guided pairs).
Original step-15
Boltz/DBMol discrepancies persist, and `old_gate_certified` remains false.
There was no optimization or new DeFoG worker call. Details are retained in
[docs51](51_defog_matched_pilot.md) and the ledger. Missing aged-out LSF
records alone are not completion evidence; these persisted run/log/artifact
records establish completion. No full preparation, benchmark, training or
new long guided run has been submitted since the prior snapshot.
