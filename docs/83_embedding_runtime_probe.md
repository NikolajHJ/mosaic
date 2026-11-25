# Direct Boltz versus frozen-VAE mapping runtime probe

User authorized29 October2026: investigate the large slowdown using historical
direct Boltz experiments as the baseline. Subsequent instructions require one
fixed step size from the **most recent Boltz experiments**, without searching
multiple sizes, and explicitly preserve the current experiment. This probe
therefore uses **LR0.0025, one candidate**, with no halving, clipping or gradient
normalization. The existing source142/143 long run and source140 VAE continuation
are not stopped or changed. Probe states are never promoted to either campaign.

## Correct historical baseline

The earlier source138 mapping-vs-relaxed comparison still used a VAE in both
arms, so its41–47s/update does not establish the requested no-VAE baseline.
Read-only inspection of Mosaic's existing JQ1 experiments identified:

| Saved no-VAE experiment | Measured updates | Fixed LR/candidates | Mean update time, excluding first10 |
| --- | ---: | --- | ---: |
| Boltz affinity only, job29809297 segment0000 | 10000 |0.0025 /1 |1.724877s |
| Boltz DBMol composite, job29809294 segment0000 |2500 |0.0025 /1 |3.515263s |

The DBMol baseline includes the affinity and independent structural/confidence
trunks, the same named composite arithmetic and coefficient family. Neither
baseline uses a VAE, decoder, learned mapping or chemistry repair. The latest
direct Boltz/DBMol continuation records retain LR0.0025 and fixed single
candidates. These saved timing JSONLs, configurations and summaries were read
with exact declared SHA verification; original failed terminal outcomes remain
failed and are not promoted by reusing their timing observations. The affinity
summary is failed; the selected first DBMol segment's own status is retained
verbatim in the baseline receipt. Runtime observations are not gradient or
binding evidence.

Historical jobs used an L40S/native-worker setup, different prepared target/
pose/context and software. Their JAX controller runs on CPU while native Torch
executes on the GPU. They cannot alone identify the VAE's causal overhead. The
new controlled probe instead compares the exact same original source142 models,
context, chemistry primals and dtypes on one diagnostic GPU, with/without the
VAE path. Attempt002 uses an L40S, matching the historical GPU class; the
ongoing production experiment continues on its A100.

Saved baseline evidence:
`outputs/hpc-results/embedding-runtime-probe/baseline-and-inputs-1793264955632463600.json`.
No Mosaic source, run, monitor, report or settings were modified or restarted.

## Inputs and isolated measurements

Source144 imports the authenticated **original source142** numerical modules,
not potentially changed working-tree model code. It uses original frozen VAE36,
the already trained bridge SHA
`c8d86551849c7f7e1de5218b7121cbecacb7f9dbc9ee11a5aea5972187a81551`,
calibration SHA
`1c436618bc6dc26dfb049908b11477afc5f2e7b291132db4577a99631f0c69f8`,
and the exact25-slot case001 context. Two points are tested: original start and
the immutable saved update277 state, SHA
`e719a222ed5fafa424807ae2eb8f09cdb9b4eef4146e6d00c93af63f6c9e6009`.
The original start, gate binding and trajectory manifest are hash bound.

Each measurement uses two warm-ups and five timed repeats, with CUDA
synchronization around the entire measured call. The repeats are measurements
at a fixed point, not a line search or additional production trajectory.

1. Direct atom/bond/charge logits through the same graph policy and native
   affinity only; then through full DBMol. Neither has a VAE path.
2. Cached mapped chemical projections detached as independent leaves, through
   the exact native injection and DBMol forward/backward. This holds the learned
   primal constant while removing the VAE/mapping derivative path.
3. Raw VAE→mapping→native forward/backward without per-call audit bookkeeping,
   compared with the original production learned callback and guarded callback.
   Original full frozen checks surround profiling; this diagnostic raw path
   is not a replacement optimizer or a new gate waiver.
4. VAE/mapping forward alone and VAE/mapping backward using the actual native
   chemical cotangents, excluding both native trunks from that backward timer.
5. Guard assessment, model/context integrity assertion and full context hashing
   alone; atomic state publication and representative synchronous journal I/O.
6. Exactly one LR0.0025 candidate after a gradient baseline, for direct logits,
   raw latent mapping and guarded latent mapping, plus guarded journal overhead.
   Ineligible fixed proposals are labeled and never trigger another candidate.

The detached/direct primals must exactly match their original learned/relaxed
counterparts; the raw learned latent derivative must exactly match the original
callback on the same diagnostic device. Attempt001 required the original A100
start certificate; attempt002 records its difference from the L40S result
without claiming a new production validation or changing any production gate. The
split native-cotangent VJP must recover the original latent derivative within
the explicit diagnostic tolerance. All models remain frozen and are checked
by full state digests before/after. No desired score trains any parameter.

An instrumented single-candidate call also records nested host wall and CUDA
event spans for VAE decoding, mapping, native affinity/structural/confidence
work, backward, context hashing, model versions, RDKit classifications,
feature-target/MSE/identity checks, scalar validation and journal fsync.
Inclusive spans overlap and must not be added. CUDA events measure stream-span
latency and can include host gaps, not kernel busy percentage. Instrumented
host timings can include GPU waits and instrumentation overhead; warmed
**uninstrumented** end-to-end medians are the primary comparison. Feature
shapes/bytes, GPU and memory use accompany the report. The original source142
precision and checkpoint policy are retained, not benchmark tuning variables.

## Immutable source and allocated execution

Attempt001 contains629 authenticated members. Archive SHA256
`009233fce84e8f4b06079334bbe53aed3e51a3cbbb42d5022f1d023abaac55b9`,
manifest SHA256
`e12f9737844fe7b7c81f2b8d78bd5858ce8048e39e54b585b1f059a261b5af85`,
plan SHA256
`f96e8e1ae42dfd52eda691a2694b6c8529ac5f6937d914ed4f00f15a19946dc5`.

CPU29846860 **DONE**, full stage15.006s: **21 passed**, zero failures/errors/
skips, pytest10.00s. Fixtures cover probe isolation, input hashes, fixed single
step policy, nested timers, exact primal/gradient matching and correct split
VJP behavior; they do not establish native timings.

GPU **29846866** was submitted for one exclusive80GB A100/four slots/1h wall,
55min internal. Its submission binds the exact CPU result SHA and zero-skip
JUnit. At initial collection it is **PEND**, with no GPU setup/result yet.
No component-time claim or VAE overhead ratio follows from submission.

### L40S replacement,29 October2026 10:41 CET

The A100 probe remained unstarted, blocked by the queue's GPU limit. The
ongoing source142 allocation's recorded maximum GPU memory was14.1GB, so a
48GB L40S provides a separate resource for this diagnostic. It also matches
the historical baseline's GPU class. No production allocation is stopped.

Attempt002 retains all numerical input pins, same-device exact primal/latent
derivative checks, fixed LR0.0025 and one candidate. Its explicit hardware
policy reports cross-device A100 certificate differences; those differences
cannot pass or waive a production scientific gate. Archive SHA256
`29c17cb7e64d9b78f9c342e309401525e247ab75d5e3613ae613eaedcf848435`,
manifest SHA256
`1f7b5c9b19f05e1077f8b1f83ae8632f0981ccc827c2d35824a53b01b9a9beec`,
plan SHA256
`3f89c017315e6537b4352c550fdac3e1ee2bfb9cf0549e681b10bade52facfbe`.

CPU **29846941 DONE**, full stage12.043s, **21 passed**, zero failures/errors/
skips, pytest7.29s. Its authenticated CPU proof binds GPU **29846982**, submitted
10:40:39 CET to `gpul40s`, one exclusive L40S/four CPU slots/1h wall and55min
internal. At10:41 CET it is **PEND**, waiting for physical GPUs on four hosts;
reserved CPU slots do not establish GPU execution. No GPU setup or timings yet.

Only the still-PEND, exact-name attempt001 diagnostic **29846866** was
terminated after proving its GPU output directory absent and replacement CPU
readiness. Its scheduler terminal state is **EXIT**, an intentional unstarted
resource replacement, not a failed numerical test. The first retirement
preflight rejected a locally mistaken CPU status spelling before writing any
intent or cancelling anything; its failed receipt is retained. The corrected
preflight used the authentic `passed` status and retained an exclusive intent
and scheduler response. Source142 **29846090 RUN** and VAE **29845931 RUN**
were independently reconfirmed. Neither campaign's code, settings or state was
modified by this probe.

Receipts are under `outputs/hpc-results/embedding-runtime-probe/` and
`outputs/hpc-results/source144-embedding-runtime-probe-attempt002/`.

The probe has its own immutable release/run/control paths and obtains fresh
quota. It neither attaches a profiler to nor sends a signal to a running job.
All model execution, fixtures and profiling are allocated DTU HPC work; local
work is source/JSON/byte inspection, packaging and evidence transfer.

Commands:

```text
py -3 scripts/embedding_runtime_campaign.py seal
py -3 scripts/embedding_runtime_campaign.py stage
py -3 scripts/embedding_runtime_campaign.py submit --stage cpu
py -3 scripts/embedding_runtime_campaign.py collect --attempt attempt002
py -3 scripts/embedding_runtime_campaign.py submit --stage gpu --attempt attempt002
```

Entry points are `scripts/hpc/embedding_runtime_probe.py`,
`tests/test_embedding_runtime_probe.py`, and
`configs/embedding_runtime_probe.json`. Preserve failed/partial attempts and
raw results. No performance change is automatically adopted into production.

### Read-only status,29 October2026 13:05 CET

GPU **29846982 remains PEND**, waiting for physical GPUs on three hosts.
Four CPU slots are reserved on `n-62-13-16`; this is not GPU execution. There
are still no GPU setup, result or log artifacts, so no component timings or
causal VAE overhead estimate exist yet. The fixed LR0.0025/single-candidate
protocol is unchanged. Saved **21/21 CPU passes** remain valid: the mixed
collection's return255 accompanies the aged-out CPU lookup, while the fresh
probe-only scheduler lookup returns0 and confirms PEND.

The existing optimization **29846090 RUN** has a latest saved case001 state
**652/10000**, persisted13:04:37 CET, and observations through650. Cases002/003
remain at their20-update pilot states. Observation650 is eligible and both
natural/conditioned graphs are strict-valid; this establishes no binding gain
or changed identity. These are saved file observations,
not exact live counters or a new molecular-improvement result. VAE continuation
**29845931 RUN** is unchanged. This check ran no numerical work and changed
no runtime, scientific setting, submission or job.

Exact collection and live scheduler/progress receipts are SHA256-bound in
`outputs/hpc-results/source144-embedding-runtime-probe-attempt002/status-summary002.json`.

### Completed GPU probe,29 October2026; collected22:12 CET

Probe **29846982 DONE** on `n-62-13-19`/L40S,16:07:28--16:27:09 CET.
Internal profiling took1,175.302s. Both original start and saved update277
passed exact same-device raw learned scores, latent derivatives and detached
native-input primals. The split native-cotangent VJP recovered the complete
latent gradient with maximum error0. All frozen model/context state checks
passed. Forty-one raw CPU/GPU JSON, parity, timing and LSF assets were downloaded
and checked against remote SHA256/size; the raw bytes are preserved in
`terminal-20261029/`.

The following are **mean seconds per fixed single-candidate update**, with
five repeats after two warm-ups. Medians are included because the protocol
uses uninstrumented full-call medians as its primary comparison.

| Full call | Original start mean / median | Saved277 mean / median |
| --- | ---: | ---: |
| Direct DBMol logits, no VAE |7.754 /7.689 |7.756 /7.692 |
| Raw latent VAE+mapping |8.012 /8.079 |8.025 /8.080 |
| Guarded latent |9.060 /9.044 |9.054 /9.049 |
| Guarded latent with journal |9.078 /9.058 |9.100 /9.099 |

All calls use LR0.0025 and exactly one candidate, without search, clipping or
normalization. The guarded candidates remained eligible at both points.
The full direct-versus-latent difference includes preparation and different
optimization variables; it is not a pure VAE compute estimate. The isolated
VAE+mapping forward took about0.0205s, and its forward plus backward using the
actual native chemical cotangents about0.049s. This measured compute is small
relative to the multi-second Boltz calls. Guarded fixed updates add about1.04s
mean over raw latent updates; journal I/O adds about0.018--0.046s. One frozen
assert costs about0.109--0.110s and context hashing about0.0325s; repeated checks
accumulate. Nested instrumented spans overlap and must not be summed.

The ongoing source142 optimizer still has its original16-trial policy. Its
earlier saved audit observed15.52 trial checks and8.92 native candidate calls
per update, in addition to the gradient baseline and periodic observations.
Those repeated evaluations and checks are substantial extra work relative to
the latest direct experiment's single candidate. No production optimizer,
scientific guard or VAE continuation was changed by the diagnostic.

The historical no-VAE DBMol mean3.515s still differs from this probe's no-VAE
7.75s. That remaining gap exists without the VAE and cannot be assigned to it;
target/context, software and execution conditions differ. LSF history also
retains a16:11:04 clocks event due to a power violation, explicitly allowing
the workload to continue. No clock traces were captured, so these absolute
times are not guaranteed clean-clock hardware benchmarks. The same-allocation
controls are useful observations; they do not prove a hardware cause for the
historical gap. Across A100 and L40S the saved start score/gradient certificates
are not bitwise equal; this is explicitly reported and is neither a new
production gate pass nor an adopted numerical change.

At22:12 CET the unchanged long run **29846090 RUN** had a latest saved case001
state**1703/10000** at22:12:26, with eligible/strict-valid observation1700.
Its decoded hard identity and complete hard scores exactly match the initial
observation: hard total loss8.3443440523, while learned loss is7.8065952631.
The learned surrogate reduction therefore has not produced a changed decoded
molecule or hard-score improvement at that observation. Cases002/003 remain
at20 pilot updates, eligible/valid with their respective initial hard identities
and hard scores unchanged. Saved states are not exact live counters, and these
fixed-context scores are not new fresh-native binding evidence.

The VAE's previous allocation is no longer live; its next allocation
**29849760 PEND** and callback**29849761 PEND** are the current scheduler
records. This collection does not claim VAE training is currently RUN and
changed no continuation settings. Exact evidence, measurements and raw-asset
hashes are bound in
`outputs/hpc-results/source144-embedding-runtime-probe-attempt002/status-summary003.json`.
