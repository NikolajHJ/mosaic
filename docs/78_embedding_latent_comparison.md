# Matched latent optimization: learned embeddings versus relaxed chemistry

## Attempt006 full CPU readiness passed; GPU queued, 27 October 2026 12:13 CET

The [actual CPU terminal snapshot](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt006/status-1793099462736059700.json)
records **29829824 DONE**, **12:10:00–12:10:50 CET** on n-62-11-63. **All 283
fixtures passed**, zero failures/errors/skips; JUnit31.595s, full runner
**41.6918s**. The declared CPU-only `mkl_compatible_four_threads_v1` profile
covered fixtures and actual readiness; native GPU arithmetic/settings are
unchanged. Original36 default decoder outputs matched bitwise and frozen
VAE state was unchanged. [Exact CPU terminal copies and integrity](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt006/terminal-cpu-20261027/integrity.json)
preserve the complete pass and readiness receipts.

All three parent starts' **latents, masks, dtypes, records and hard identities
matched exactly**. `starts.pt` retained the parent's SHA256
**`832592de7183b51ac22542ead558bc453006d3d349d942e2933dd1a52fb40ebd`**.
The new final-policy FP64/cast-back calculation changed probabilities by at
most **`1.7881393432617188e-7`**, with hard identities preserved. This is CPU
candidate readiness, **not a native correction pass**; all fresh native and
derivative/production-binding gates remain mandatory before optimization.

The [guarded GPU submission](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt006/submit-gpu-1793099470162269900.json)
returned zero: **29829828 submitted at 12:11:13 CET**, one exclusive80GB A100,
four slots, 24-hour walltime/23-hour internal cap, `submitted_not_passed`.
Fresh quota was **47.01/100 GiB used, 52.99 GiB free**. The
[12:11:25 scheduler/artifact snapshot](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt006/status-1793099485493670200.json)
confirms **PEND**, a new job waiting for scheduling, with no GPU artifacts,
native gates or optimization updates. The [latest 12:13:50 read](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt006/status-1793099630900207300.json)
still records **PEND**, now reporting job-slot limits on four hosts, with
157 eligible seconds and zero ineligible seconds. There is no reliable ETA. All earlier failures
remain preserved; independent VAE/epoch45 work is unchanged by this comparison.

## CPU kernel controls completed; profiled attempt006 submitted

The [complete kernel-probe002 result](../outputs/hpc-submissions/source138-embedding-cpu-kernel-probe002/terminal/result.json)
records **29829777 DONE**, **27 October 12:04:00–12:07:01 CET** on n-62-29-11.
The unchanged 268-fixture suite reproduced both inherited exactness failures
with four baseline threads and with one baseline thread. With
**`MKL_CBWR=COMPATIBLE` and four threads, all 268 passed** in 47.96s; portable
one-thread mode with the same MKL setting and `ATEN_CPU_CAPABILITY=default`
also passed all 268 in 36.59s. All modes retained zero errors/skips and unchanged
assertions. [Exact terminal logs/JUnit integrity](../outputs/hpc-submissions/source138-embedding-cpu-kernel-probe002/terminal/integrity.json)
preserves both failed controls and successful runtime modes. This remains a
kernel diagnostic, not full readiness or a native GPU pass.

The original pinned probe001 **29829748** was still pending and was canceled
as redundant only after checking its exact job ID/name;
[its status/cancellation receipt](../outputs/hpc-submissions/source138-embedding-cpu-kernel-probe001/redundant-probe-status.json)
is preserved. No independent VAE or other campaign job was changed.

New independent
`source138-epoch36-embedding-comparison-attempt006`
[seal](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt006/seal.json)
contains **603 members**; upload/authenticated staging returned zero. Its
[actual CPU submission](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt006/submit-cpu-1793099397004767200.json)
returned zero, submitting **29829824**, four slots, `rusage[mem=8GB]`, one host,
45 minutes, initially `submitted_not_passed`. At this preparation snapshot,
full readiness and all **283 expected fixtures**, including 15 new profile
fixtures, were pending and no GPU was submitted. The completed readiness and
subsequent guarded GPU submission above supersede that initial status.

| Attempt006 binding | SHA256 |
| --- | --- |
| Source archive | `456c8cc8ae0fbc2d84dcdadddd7734f879cefc9db4204275b8bcd0525ec62acb` |
| Release manifest | `aedc247e662a9faa974f90f55173f383c3b93128dbab9c5a7d15681b739919b8` |
| Comparison plan | `58b47026f175622992c8ea1e257e02d1836387c752aa024bb2fcc0201e886802` |

The explicit **`mkl_compatible_four_threads_v1` CPU-only profile** is applied
before importing torch to both fixture execution and actual original-model
parity/shared-start preparation. It preserves every assertion and tolerance and
changes no native GPU setting. The [static protected-source audit](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt006/static-audit.json)
confirms shared VAE model, graph policy, representation, native/bridge and
optimizer bytes unchanged. The broad archive also includes unused concurrent
chemistry epoch45/continuation files and earlier Protenix changes; this chat did
not modify that independent work. All failed attempts and baseline controls
remain preserved; no native correction or optimization success follows from
this CPU submission.

## Attempt005 CPU gate failed; unchanged-assertion kernel diagnostic queued

The [terminal attempt005 snapshot](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt005/status-1793098651622031600.json)
records **CPU 29829719 EXIT1 at 27 October 11:57:15 CET**, n-62-28-35.
**266/268 fixtures passed**, two failures, zero errors/skips, pytest34.14s /
JUnit34.036s. **All 45 new correction/policy fixtures passed**. The two
inherited failures were raw decoder bond-head **bitwise symmetry** and bridge
atom-head **exact permutation equality**, the latter's maximum discrepancy
`5.960464477539063e-8`. Their strict assertions and failed evidence are preserved.
Attempt005's full CPU readiness **did not pass** and stopped before actual
original36 loading/shared-start preparation; **no GPU was released**.

A separate [kernel diagnostic submission](../outputs/hpc-submissions/source138-embedding-cpu-kernel-probe001/submit.json)
returned zero: **29829748**, pinned to the same failed host **n-62-28-35**,
four CPU slots, ten minutes. It authenticates the unchanged attempt005 release
and reruns the full **268-fixture suite without assertion changes** under three
modes: baseline four threads, four threads with `MKL_CBWR=COMPATIBLE`, and
baseline one thread. The [first status receipt](../outputs/hpc-submissions/source138-embedding-cpu-kernel-probe001/status001.json)
records **PEND**, with the host reporting a recently started job and no result
yet. This is **diagnostic only**, not a CPU readiness pass or permission to
promote a failed attempt. No model, fixture tolerance, scientific setting or
VAE continuation was changed.

## Corrected retry attempt005 authenticated; CPU running, 27 October 2026

The new independent
`source138-epoch36-embedding-comparison-attempt005`
[seal](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt005/seal.json)
contains **596 members**; upload/authenticated staging returned zero. The
[actual CPU submission](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt005/submit-cpu-1793098587111580600.json)
returned zero and submitted **29829719 at 11:56:29 CET**: four slots,
`rusage[mem=8GB]`, one host, 45-minute walltime, `submitted_not_passed`.
The [saved scheduler/artifact snapshot](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt005/status-1793098606989962400.json)
confirms **RUN since 11:56:30 on n-62-28-35**. CPU readiness remains pending;
**no GPU is submitted and no numerical correction pass is established**.
Fresh submission quota reported **53.94 GiB free**.

| Attempt005 binding | SHA256 |
| --- | --- |
| Source archive | `2de705ee8d88c4699d1f12c105553b1111bf7a82c49e88611750339c9d2e6be1` |
| Release manifest | `83fafc94636921348d140ce1b7bbb6bb4c6dca57e6224e90ed7095d8d5519975` |
| Corrected comparison plan | `3b26ae0dd1303d5a39cd42ef79f59fa088380dcc867ac57db9ee91cc85e8323d` |

The [static audit](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt005/static-audit.json)
confirms unchanged shared graph policy, representation, adapter, native models,
objective, bridge and production optimizer. Only the comparison uses the new
final-policy precision candidate; the broad archive also snapshots unused
concurrent Protenix changes. Static byte checks do not establish native success.

[Attempt004's preserved not-promoted receipt](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt004/not-promoted.json)
records a locally sealed candidate that was **never staged, submitted or
promoted**. Final review added explicit equality of the original parent `z` and
active-mask dtypes in readiness and two fixtures, so a new immutable attempt005
was used. **Attempt004 had no numerical failure.** All earlier actual failures
remain preserved; the same original36/bridge/three starts/settings/tolerances
and fresh full gates remain mandatory. VAE continuation is unchanged by this
work.

## Diagnostic completed; final-policy precision candidate selected, 27 October 2026

The [terminal snapshot](../outputs/hpc-results/source139-epoch36-embedding-gradient-diagnostic-attempt001/status-1793097916707003400.json)
confirms **GPU 29821249 DONE**, **26 October 23:31:48–23:34:47 CET** on
n-62-18-12, 179s allocated/159.643s runner elapsed. Its live scheduler record
has aged out; saved LSF completion and the structured result establish success
of the diagnostic execution, `completed_diagnostic_no_fix_selected`. No
optimization ran.

Complete terminal copies were authenticated against the original remote sizes
and hashes in the [integrity receipt](../outputs/hpc-results/source139-epoch36-embedding-gradient-diagnostic-attempt001/terminal-20261027/integrity.json):

| Verified terminal artifact | SHA256 |
| --- | --- |
| [Native diagnostic, 40,418 bytes](../outputs/hpc-results/source139-epoch36-embedding-gradient-diagnostic-attempt001/terminal-20261027/gpu-diagnostic.json) | `be2403a9616a3345791e748747b9e2ca99fb55a0bab5641e506a473b2e48d9f9` |
| [GPU result, 40,727 bytes](../outputs/hpc-results/source139-epoch36-embedding-gradient-diagnostic-attempt001/terminal-20261027/gpu-result.json) | `e3cbd22abe80329424c8484592c734881d9d2df3c135701acc2ad4d8cd537f36` |

The diagnostic reproduced case003's exact parent **scores, production gradient
and tested direction**. Full-forward values and every cut's primal tensors
matched exactly. The decoder, graph-policy and adapter replays retained the
complete captured cotangents: each replay's full input VJP tensor matched the
captured production VJP bitwise, not merely its scalar contraction. Native
oracle validation passed; frozen VAE, scorer, projection bank, point and context
identities/versions remained unchanged.

For the total-loss directional contraction, the local forward/reverse errors
were:

| Replayed boundary | Absolute error |
| --- | ---: |
| Decoder | `4.641983438261055e-10` |
| Final relaxed-graph policy | `3.4772080571556993e-6` |
| Native-chemistry adapter | `5.792675367155553e-15` |

This localizes the reproduced mismatch to the final graph-policy calculation
for this state/direction. The original full-path error and failed decision
remain unchanged. A control evaluating **only the final policy in float64,
then casting its outputs back**, reduced the fixed-cotangent total error to
**`5.071365666818595e-10`**. It held the original native graph cotangent fixed;
no native score was evaluated at the slightly changed primal. Maximum graph
changes were about 1.19e-7 for atoms and 5.96e-8 for bonds. This control is **not
a passed native correction or full comparison gate**. The uncheckpointed
64-block scorer control was explicitly not run because it exceeded the bounded
80GB diagnostic design; complete captured-cut replays supplied the attribution.

Under the standing correction/resubmission authorization, the selected smallest
candidate is **comparison-specific float64 arithmetic for the final graph
policy, followed by a cast back to the original dtype**. Shared VAE/bridge/native
models, default VAE training/decoding and the independent continuation remain
unchanged. Preserve the exact same three starts, objective, update/step settings,
budgets and tolerances; fresh complete native/derivative/production-binding
gates for all starts must pass before optimization. Comparison attempt004 is
**not yet sealed or submitted**, and the candidate has no native pass yet.

## Access recovered; diagnostic still queued, 26 October 2026 22:39 CET

The [fresh successful read](../outputs/hpc-results/source139-epoch36-embedding-gradient-diagnostic-attempt001/status-1793050788490890700.json)
confirms that SSH access has recovered. **GPU 29821249 is still PEND**:
four CPU slots are reserved on **n-62-18-8**, while physical GPU availability
(`ngpus_physical`) blocks execution. Reservation is not execution. No GPU
launch artifacts, logs or native diagnostic result exists, and there is no
reliable start estimate. The CPU scheduler record has aged out; its saved
full-stage pass, **223 fixture passes** and original36 parity remain valid.
The numerical correction and full comparison retry still await native evidence.
This read changed no runtime code, settings, jobs or VAE continuation.

## Later collection blocked by SSH timeouts; last-known GPU status remains pending

The [last successful snapshot, 26 October 13:52:49 CET](../outputs/hpc-results/source139-epoch36-embedding-gradient-diagnostic-attempt001/status-1793019169106562100.json)
records diagnostic **GPU 29821249 PEND**, waiting for `ngpus_physical` on one
host, with no GPU artifacts or result. Later normal read-only collections
failed with SSH255/timeouts to login1:
[first receipt](../outputs/hpc-submissions/source139-epoch36-embedding-gradient-diagnostic-attempt001/collect-gpu-1793019372650144900.json),
[second receipt](../outputs/hpc-submissions/source139-epoch36-embedding-gradient-diagnostic-attempt001/collect-gpu-1793019410280427500.json).
These are access failures, **not evidence of GPU failure or a new job state**.

A [verified alternate-login attempt](../outputs/hpc-submissions/source139-epoch36-embedding-gradient-diagnostic-attempt001/alternate-login-verification.json)
also timed out. Its host key was checked against DTU's official fingerprint
and stored in a task-local public `known_hosts` file; global SSH configuration
was unchanged and authentication was not bypassed. No reliable current start
estimate or native diagnostic outcome is established.

The user's correction/resubmission authorization remains active. The next
dependent step is to read the native diagnostic evidence, implement the smallest
demonstrated correction and rerun the full gates. No numerical fix or comparison
attempt004 is selected yet; all failures and independent VAE work are preserved.

## Diagnostic CPU passed; GPU queued, 26 October 2026 13:50 CET

The [CPU terminal snapshot](../outputs/hpc-results/source139-epoch36-embedding-gradient-diagnostic-attempt001/status-1793018959232362600.json)
records **29821224 DONE**, 13:47:52–13:48:43 CET, n-62-30-23. **All 223
fixtures passed**, zero failures/errors/skips, JUnit32.285s; the complete stage
took **41.000s**. Original36 decoder parity, unchanged frozen state and
reproduction of all three prescribed parent starts passed.

The actual CPU decoder-only duality probe had absolute error
**`1.5692636872219985e-8`**, within inherited bound
**`1.1245987208140518e-6`**. It used deterministic engineering cotangents,
**not native Boltz cotangents**; it is diagnostic attribution evidence, not a
passed native derivative, selected fix or promotion of the failed historical
gate. Frozen VAE state remained unchanged.

The [exact-release CPU-guarded GPU submission](../outputs/hpc-submissions/source139-epoch36-embedding-gradient-diagnostic-attempt001/submit-gpu-1793018964838402000.json)
returned zero: **29821249 submitted 13:49:29 CET**, one exclusive80GB A100/four
CPU slots, one-hour walltime/55-minute internal cap. Fresh quota was
**48.20/100 GiB used, 51.80 GiB free**. The
[13:50:06 saved scheduler snapshot](../outputs/hpc-results/source139-epoch36-embedding-gradient-diagnostic-attempt001/status-1793019006060069300.json)
confirms **PEND**, waiting for `ngpus_physical` on one host. No reliable ETA or
GPU artifacts/result exists. No numerical correction or comparison attempt004
is selected/submitted; no optimization ran and independent VAE continuation
remains unchanged.

## Diagnostic attempt001 authenticated; CPU running, 26 October 2026

The new independent
`source139-epoch36-embedding-gradient-diagnostic-attempt001`
[seal](../outputs/hpc-submissions/source139-epoch36-embedding-gradient-diagnostic-attempt001/seal.json)
contains **592 members**. Upload and authenticated staging returned zero. The
[actual CPU submission](../outputs/hpc-submissions/source139-epoch36-embedding-gradient-diagnostic-attempt001/submit-cpu-1793018868370240500.json)
returned zero and submitted **29821224 at 13:47:51 CET**: four slots,
`rusage[mem=8GB]`, single host, 45-minute walltime. The
[saved scheduler/artifact snapshot](../outputs/hpc-results/source139-epoch36-embedding-gradient-diagnostic-attempt001/status-1793018900321656700.json)
records **RUN since 13:47:52 on n-62-30-23**. CPU fixture/model readiness is
pending; no pass or diagnostic GPU submission is established in this snapshot.

| Diagnostic attempt001 binding | SHA256 |
| --- | --- |
| Source archive | `bd677171c8890c888191e1280ed62c8b84bb622d6b8e8b0f13cdbe5656ac8740` |
| Release manifest | `90f744e73a71dc79a9165c2db33353c096d0484bde5398ba7c95386eda547ff9` |
| Diagnostic plan | `517d1086d9eb69c9ea101cb1125f8d87e31ee2341fc6e053b5e989467ac3df7a` |

The [static member comparison](../outputs/implementation-checks/source139-embedding-diagnostic-vs-comparison003-manifest.json)
confirms protected VAE/native/bridge and production-optimizer modules are
unchanged. This broad snapshot also contains unrelated concurrent Protenix
changes; it is not an archive differing only in the diagnostic files. Static
byte checks do not establish numerical success. No optimization, selected fix
or comparison attempt004 follows from this CPU submission; independent VAE
continuation and all earlier failed attempts remain unchanged.

## Targeted correction and resubmission authorized, 26 October 2026

The user authorized **“fix it and resubmit”** after attempt003's relaxed-path
derivative failure. A new immutable
`source139-epoch36-embedding-gradient-diagnostic-attempt001` is being prepared
to localize the mismatch before selecting a correction. Allocated checks will
split the full derivative at the decoder-logit, relaxed-graph and native-chemistry
boundaries. CPU readiness precedes a bounded **one-hour GPU allocation with a
55-minute internal cap**; this diagnostic performs **no optimization updates**.

Retain original epoch36, the trained bridge, exact prescribed starting latents,
full objective, budgets, tolerances and every failed record. The authorization
does not permit automatic gate weakening or dropping a start. No root cause,
chosen correction, passed diagnostic or submission is established by this
preparation note. A corrected comparison attempt004 is **not yet selected or
submitted**; the VAE continuation remains unchanged.

## Terminal result collected 26 October 2026: derivative gate stopped optimization

The [26 October 12:52 CET read-only snapshot](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt003/status-1793015544565816000.json)
supersedes the pending observations: **GPU 29802743 ran on n-62-18-12 on
25 October, 06:33:02–06:48:42 CET, then EXIT1**. Saved LSF output establishes
the exit; the live scheduler record has aged out. Runner elapsed time was
**919.304s**, versus **940s** in the scheduler's allocation summary.

All **three native-oracle checks passed**. Cases001/002 passed both relaxed
and learned start gates and exact production-gradient bindings. **Case003's
relaxed path failed** independent forward/reverse AD agreement in the gradient
direction; its three random directions passed. The mandatory failures were:

| Component | Absolute error | Allowed error |
| --- | ---: | ---: |
| Affinity loss | `2.5204821007445213e-6` | `2.0817236646993208e-6` |
| Total loss | `3.4778529073341627e-6` | `2.583432959841326e-6` |

Contact/confidence comparisons in that direction passed, and frozen-module
and context checks passed. The generic exception says “Learned-start”, but the
[actual failed gate](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt003/terminal-gates-1793015579941148800/case-003/relaxed-start-gate.json)
records `scoring_path=relaxed`; **case003's learned gate never ran**. This does
not establish a root cause or a validated fix.

All prescribed starts had to pass before trajectories. There are **no
optimization directories, no fresh-native stage and zero optimization
updates**; no molecular-performance comparison exists. The
[complete gate summary](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt003/terminal-gates-1793015579941148800/summary.json)
and [exact-byte integrity receipt](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt003/terminal-gates-1793015579941148800/lf-normalization-integrity.json)
preserve all 12 original remote gate/result files, 890,446 bytes. Earlier
collector-truncated/local newline-translation evidence remains preserved and
is superseded by these exact copies. This status collection changed no runtime
code, settings, jobs or independent VAE continuation.

## Read-only status, 24 October 2026 10:54 CEST

The [fresh saved snapshot](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt003/status-1792832067824362000.json)
confirms **GPU 29802743 remains PEND**, now reporting job-slot limits on
**four hosts**. Eligible pending time is **47,901 seconds** (about **13 hours
18 minutes**), with zero ineligible time. No GPU logs, gates, result or
optimization progress artifacts exist, and there is no reliable start ETA.
The saved full CPU pass, all 212 fixture passes and authenticated three shared
starts remain valid. This read-only check changed no runtime code, settings
or jobs and left the independent VAE continuation unchanged.

## Read-only status, 24 October 2026 01:30 CEST

The [fresh scheduler and artifact snapshot](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt003/status-1792798218154998400.json)
confirms **GPU 29802743 is still PEND**, with job-slot limits on five hosts.
Eligible pending time is **14,051 seconds** (about 3 hours 54 minutes), with
zero ineligible pending time. There is no reliable start estimate and no GPU
log, result, derivative gate or optimization progress artifact.

The CPU scheduler record has aged out; its saved full-stage `passed` result,
all **212 fixture passes** without failures/errors/skips, and authenticated
shared starting latents remain available and valid historical evidence. This
status check changed no runtime code, settings or jobs, and left the independent
VAE continuation unchanged.

## Attempt003 CPU readiness passed; native comparison GPU queued

The [allocated CPU terminal snapshot](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt003/status-1792784006917505500.json)
records **29802735 DONE**, **23 October 2026 21:32:09–21:32:49 DTU-local**,
n-62-31-20. **All 212 fixtures passed**, zero failures/errors/skips, JUnit
25.020s; the complete CPU/model/start stage took **32.1028s** and returned
`passed`. Strict original36 loading, bitwise ordinary/capture decoder parity,
unchanged frozen state and actual shared-start preparation passed. This is
allocated CPU readiness; it does not pass native GPU derivatives or molecular
optimization.

The prescribed starting samples were **indices 0, 3 and 6**, the first audit
posterior-mean groups with distinct decoded identities in authenticated stored
order. Their occupied counts are **25, 19 and 19**. Each pair shares the same
saved latent and bound native context; no start was replaced according to a
score or gradient. The immutable `starts.pt` SHA256 is
`832592de7183b51ac22542ead558bc453006d3d349d942e2933dd1a52fb40ebd`;
CPU JUnit SHA256 is
`613c8e0b193e1e6ef7907615cd93f198a15417497f2fec3e30818f82606ec3b1`.

The [actual GPU submission](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt003/submit-gpu-1792784164515503800.json)
returned zero, **29802743 submitted at 21:36:07 DTU-local**,
`submitted_not_passed`: `gpua100`, **one exclusive 80GB A100/four CPU slots**,
`rusage[mem=16GB]`, **24-hour walltime and 23-hour shared internal cap**.
Fresh submission quota was **64.80/100 GiB used, 35.20 GiB free**.

The [saved 21:37:42 queue snapshot](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt003/status-1792784262997389600.json)
confirms **29802743 PEND**, job-slot limits on **five hosts**, eligible pending
time **96s**, ineligible **0s**. It supersedes the initial
[21:36:20 new-job wait](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt003/status-1792784180809306300.json).
No GPU logs/result artifacts, native derivative
pass, trajectory start or fresh native outcome exists in this observation.
There is no reliable start estimate. Submission leaves the VAE continuation
unchanged; its similarly numbered independent jobs are a different campaign.

## Exact context bank restored; unchanged attempt003 CPU submitted

The [guarded remote restoration completion](../outputs/hpc-submissions/source138-context-restore-20261023/completion.json)
at **23 October 2026 21:30:43 DTU-local** records `exact_context_bank_restored`:
all **44 cold-offloaded source056 prepared contexts**, **1,451,429,736 raw
bytes**, were restored to their original paths with exact recorded sizes and
SHA256. The [transport/restoration receipt](../outputs/hpc-submissions/source138-context-restore-20261023/remote-restoration.json)
and durable intent/completion preserve the operation. No model was regenerated,
job or setting changed, checkpoint written or original cold archive modified.

The [independent read-only verification](../outputs/hpc-submissions/source138-context-restore-20261023/independent-verification.json)
at **21:31:52 DTU-local** rehashed **all 50 catalogued contexts**, including the
six contexts that had remained online, and passed. Original epoch36, trained
bridge and source136 immutable archive hashes matched before and after
restoration and in this independent check. Immediate post-restoration quota
was **64.41/100 GiB used, 35.59 GiB free**; the later submission snapshot is
different and includes concurrent storage activity.

New independent
`source138-epoch36-embedding-comparison-attempt003`
[seal](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt003/seal.json)
contains **582 members**, with **the exact same member-manifest and comparison
plan hashes as attempt002**. No source change was needed to restore the missing
inputs. Its upload/authenticated staging and
[actual CPU submission](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt003/submit-cpu-1792783925232386600.json)
returned zero, submitting **CPU 29802735**, four slots, `rusage[mem=8GB]`,
one host, 45-minute walltime, `submitted_not_passed`. Fresh submission quota was
**64.17/100 GiB used, 35.83 GiB free**. The submission alone establishes no
CPU pass, GPU job, derivative result or optimization start. Both failed earlier
attempts retain their sealed bytes and outcomes.

| Attempt003 binding | SHA256 |
| --- | --- |
| Source archive | `bccba5d6a762349b450baf03b24251e12994be598926ce8960f6e5869d55ce9b` |
| Unchanged release manifest | `4a598d1143f6afc180d18dea618cdc09a859f019a7fc042b9025fcb8fb8487dd` |
| Unchanged comparison plan | `60f555eb6193b804d8446dc864a57c52662e4c146be424b8f7cc118f6d6ecd6c` |

## Attempt002 fixtures passed; start preparation requires exact context restoration

The [attempt002 terminal snapshot](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt002/status-1792783426704147500.json)
records **CPU 29802683 EXIT, exit 1**, **23 October 2026 21:22:38–21:23:17
DTU-local**, n-62-31-16. Its numerical fixture gate **passed all 212 cases**,
zero failures/errors/skips, pytest 25.64s/JUnit 25.552s. Strict loading of
authenticated original epoch36, bitwise ordinary/capture decoder parity and
unchanged frozen tensor state also passed. JUnit SHA256 is
`3216abfe7fe65bbe69a2171afd4f88376da054941f2c930a6e7b1c9531fa3991`.

The **whole CPU stage did not pass**: actual start preparation then raised
`FileNotFoundError` for historical source056
`contexts/count-19/prepared.pt`. No shared-start artifact or GPU submission
followed. This is a missing cold-offloaded input, not a failing fixture, native
gradient or new chemical outcome. Preserve the complete failed attempt.

The [verified 21 October cold-offload restoration map](../outputs/hpc-results/storage-cleanup-20261021-001/cold2-restoration-map.json)
contains the exact missing context and the other 43 offloaded source056 count
contexts. Its original archive is preserved locally at
`outputs/hpc-results/storage-cleanup-20261021-001/cold2-payloads.tar.gz`, SHA256
`ec6797d064f5219b1279ea12b92a737a314bbfd02ebd8c3be25c690968ce866e`.
Restoration preparation uses standard-library byte extraction and verification
for all **44 source056 prepared contexts**, without regenerating models,
altering the prescribed starts or selecting only favorable counts. Exact
restoration evidence is recorded separately when complete; no restoration or
replacement allocation follows merely from locating that archive.

The later [local extraction proof](../outputs/hpc-submissions/source138-context-restore-20261023/local-extraction-proof.json)
now establishes **all44 exact source056 contexts** extracted/authenticated
against the cold-offload map, **1,451,429,736 raw bytes**. The original cold
archive hash matched before and after extraction and remains unchanged. The new
restore capsule has exactly **45 regular members**: those44 files under their
original paths relative to `/work3/s234818/latentmol`, plus its manifest. No
checkpoint or other cold payload is included; no model was imported and no
remote path or job changed during this preparation.

| Restoration preparation artifact | SHA256 |
| --- | --- |
| [18,476,907-byte restore capsule](../outputs/hpc-submissions/source138-context-restore-20261023/source056-prepared-contexts-exact-restore.tar.gz) | `ef82e950d0f1bb426d098592ba7d4b7a0584ef75fb8029babe93a57690c50d8b` |
| [Exact path/member manifest](../outputs/hpc-submissions/source138-context-restore-20261023/restore-manifest.json) | `60765caddc95f881ed2207b12ac8ed69398f9b47c3635d7c71eae08cd32938e1` |

This is a verified local restoration package. Guarded upload/remote restoration,
fresh quota and an independent allocated retry must be established by their own
receipts before any new readiness or optimization claim.

Attempt002's [seal](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt002/seal.json)
contains **582 members**; upload/staging and its
[CPU submission](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt002/submit-cpu-1792783353627483400.json)
returned zero. Fresh submission quota was **63.07/100 GiB used, 36.93 GiB free**.

| Attempt002 binding | SHA256 |
| --- | --- |
| Source archive | `b3041c26afa15f10983c9746813885e72b0e1bbc35326c7f85d5002b3ca07d06` |
| Release manifest | `4a598d1143f6afc180d18dea618cdc09a859f019a7fc042b9025fcb8fb8487dd` |
| Unchanged comparison plan | `60f555eb6193b804d8446dc864a57c52662e4c146be424b8f7cc118f6d6ecd6c` |

Release and campaign are the attempt001 paths below with terminal
`attempt002`. All frozen VAE/bridge/Boltz bindings and independent continuation
remain unchanged. A future independent attempt must complete CPU/model/start
preparation and all native derivative checks before matched trajectories.

## Attempt001 failed two fixture hash-path calls; evidence preserved

The [attempt001 terminal snapshot](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt001/status-1792783307992603600.json)
supersedes its initial submission-only state: **CPU 29802658 EXIT, exit 1**,
**21:20:59–21:21:33 DTU-local**, n-62-30-24. Of **212 cases**, **210 passed
and two failed**, zero errors/skips, pytest 24.49s/JUnit 24.387s. Both failures
were fixture calls in `tests/test_embedding_comparison.py` that passed strings
to the existing `file_sha256(Path)` helper while checking saved checkpoint
hashes. The gate stopped before original-model/start preparation; no attempt001
GPU was submitted. No failure is relabelled as passed.

The comparison-specific correction imports `Path` and wraps those two fixture
arguments. A [static sealed-manifest comparison](../outputs/implementation-checks/source138-embedding-attempt001-002-manifest-delta.json)
confirms all `src/` members and the comparison plan are identical, with no
added/removed members. The two changed archive members are the corrected test
and `scripts/hpc/check_chemistry_launch.py`, the independently edited VAE
startup-correction helper that this comparison runner/operator does not call.
The comparison's production code, scientific settings and gates are unchanged;
this byte evidence is separate from allocated numerical validation.

## Attempt001 sealed, authenticated and CPU submitted

On **23 October 2026**, the independent
`source138-epoch36-embedding-comparison-attempt001`
[seal](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt001/seal.json)
packaged **582 authenticated members**. The
[upload](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt001/upload.json)
and [staging](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt001/stage.json)
returned zero; staging reports `staged_authenticated`.

The [actual CPU submission](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt001/submit-cpu-1792783254961755100.json)
returned zero and submitted **29802658** to `hpc`: **four CPU slots**, declared
`rusage[mem=8GB]`, single host and **45-minute walltime**. Its saved state is
`submitted_not_passed`. Fresh submission quota was **62.55/100 GiB used,
37.45 GiB free**. No CPU result, GPU submission, derivative pass, trajectory
update or fresh native outcome is established by this submission receipt.

| Attempt001 binding | SHA256 |
| --- | --- |
| Source archive | `e4764c6c6fb3444fd5032285078f5d96f4daa8c8547a5aae84dd5fedf7b69da7` |
| Release manifest | `c239d5fecc97133ea5c78aa8a7e81ac7ebab63acf238adfb7c4dc8625f440303` |
| Comparison plan | `60f555eb6193b804d8446dc864a57c52662e4c146be424b8f7cc118f6d6ecd6c` |

Release:
`/work3/s234818/latentmol/releases/source-138-epoch36-embedding-comparison-attempt001`.
Campaign:
`/work3/s234818/latentmol/runs/source138-epoch36-embedding-comparison-attempt001`.
The exact operator command binds archive/plan hashes, release and campaign;
its allocated wrapper uses `env/boltz311` through `scripts/hpc/boltz_run.sh`.
Numerical CPU tests and actual original-36 starting-data reconstruction must
pass before any dependent GPU stage. The GPU request is prospectively bounded
to **one exclusive 80GB A100/four slots, 24-hour walltime/23-hour shared internal
cap**; it is not submitted by the CPU receipt.

The source137 mapping pilot, earlier attempts and source136 VAE continuation
remain unchanged. This embedding-comparison namespace is independent of any
separately authorized VAE startup-correction candidate with a similar source
number; full release/campaign identities distinguish the studies.

## Scope and current status

On **23 October 2026**, the user authorized a matched latent-optimization test
of the completed embedding mapping against the original relaxed-molecule path,
using the **same frozen original epoch-36 VAE and identical starting latents**.
The existing source136 VAE continuation, jobs, settings, checkpoint lineage and
immutable release remain unchanged. This is an independent experiment on
`codex/epoch36-boltz-embedding-bridge`.

The implementation, allocated CPU readiness and GPU submission are established
above. The rest of this record defines the prospective scientific protocol;
GPU submission does not establish derivative, trajectory, native-score or
comparative success. The sealed plan and allocated receipts are authoritative
about what actually executes.

The preceding [mapping pilot](76_epoch36_boltz_embedding_bridge.md) completed
1,000 supervised updates and its native exact-oracle injection check. That
result is the prerequisite artifact, rather than evidence that the learned
gradient improves decoded molecules. Audit examples are correlated and charged
categories were uncovered; this comparison must retain those limitations.

| Frozen artifact | SHA256 |
| --- | --- |
| Original epoch-36 VAE | `3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f` |
| Trained mapping checkpoint | `013834958d52f7aee999ca4c39496e14fc5333d69eb697d47433821f48290343` |
| Boltz affinity checkpoint | `dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e` |
| Boltz structure/confidence checkpoint | `090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1` |
| Pilot's prepared 13-slot reference complex | `8b6e47fdf5a53a8bd92c5ebf0baf7471ca6a329bbc524e9957a3e94a9938f611` |

No VAE, bridge, projection bank or Boltz parameter is trainable during this
test. The mapping is loaded strictly and authenticated against the completed
pilot. It receives no additional reward fitting or supervised refinement.

## What the paired arms compare

Both arms minimize the same declared full DBMol scalar:

`affinity loss + 2.5 × contact loss + 2.5 × confidence loss`.

The [loss reference](77_dbmol_boltz_loss_implementation.md) defines the native
outputs, reductions, two checkpoint-specific trunks and float64 contact/loss
arithmetic. Within each case, identical pose, receptor/MSA context, pocket
residues, atom/token maps, temperature, occupancy, checkpoint precision and
objective weights are used for the paired arms. Cases can have different
occupied atom counts; each uses its own authenticated count-specific prepared
complex and the same fixed slot binding in both arms.

The **relaxed arm** follows the existing frozen decoder's atom/charge/bond
probabilities through the deterministic native-feature adapter. The **learned
arm** maps the same decoder's final hidden states into the corresponding
checkpoint-specific early chemical projections. The ordinary molecular heads
still define the learned arm's decoded hard identity. This is a comparison of
two continuous optimization surfaces associated with the same decoded graph
convention; the learned embedding need not equal an exact native molecule at
intermediate continuous points.

Occupancy and pose are fixed. No atom birth/death, latent inversion, molecule
repair, DeFoG guidance or pose gradient is part of this comparison. A fixed-slot
conditioned hard graph and its ordinary unconstrained hard decode must both be
recorded, so a changed natural occupancy cannot be hidden by the scorer's mask.
Unsupported, invalid, unchanged and charged outcomes remain in the denominator.

## Starts, update budget and step receipts

Select the **first three audit source groups with distinct decoded identities**
whose accepted saved sample has origin `posterior_mean` in the authenticated
source137 paired data, using deterministic stored order. Save their existing latent tensors once
before either arm runs, with their group/sample identities, source split,
hard labels and tensor hashes. Do not regenerate them, add perturbations,
choose a replacement according to a model score or screen for a favorable
gradient. The audit groups were held out from mapping fitting/development
selection; this does not make them independent of VAE training.

Both arms consume exact copies of these tensors, with the same dtype and
tensor hash, and request **50 updates per start**. The common settings are
learning rate **400**, maximum gradient norm **1**, trust radius **3**,
temperature **1** and `best_trial_v2` with **16 trials**. These are a bounded
paired pilot, not an estimate of general molecular-design performance. The
sealed configuration records the exact update/evaluation budgets.

A saturated or nearly flat relaxed path is a measured outcome. Starts cannot
be removed, replaced or selected according to which arm has a larger gradient
or lower Boltz score. Initial gradient norms and directions, latent movement,
hard identity changes and actual oracle cost make this outcome interpretable.

Both arms use the same declared learning rate, trust radius and
[`best_trial_v2`](62_full_budget_best_trial_optimization.md). Each state first
tries the full proposal and applies it on observed nonincrease, including
equality. Otherwise it halves through at most 16 sizes and takes the first
strict improvement, or the lowest-loss finite trial if no size improves. Such
a fallback can increase the loss. Ordinary failure to descend does not end the
requested budget. A finite zero/no-op proposal can consume an update and must
remain visible.

Save each applied state and all trial receipts. Count attempted/applied,
strictly improving, full-step equality/nonincrease, forced fallback and no-op
updates separately. Retain the terminal state and best observed state under
each arm's own scalar with their exact objective/context identities. An arm's
best learned score and another arm's best relaxed score are not interchangeable
ranking criteria.

## Derivatives and cross-scoring

Fresh allocated checks authenticate all frozen files and contexts before any
trajectory. The existing native exact-oracle injection result does not certify
the learned mapping's latent derivative. New checks must establish the learned
graph's attachment, finite supported gradients and declared multi-scale
finite-difference behavior, with observed native repeat precision retained.
Failed gates stop dependent trajectories and remain scientific outcomes.

At the predeclared checkpoints, save each continuous arm's own loss, all three
DBMol components and native outputs, latent displacement, conditioned/natural
hard atom/charge/bond tensors, validity and canonical identity. Saved `z` and
the authenticated frozen model reproduce decoder probabilities; those
probability tensors are not themselves stored by this audit. Evaluate
both continuous paths at the same saved latent when reporting gradient or loss
differences; paired starts alone do not make the later states identical.

Every valid selected hard graph receives a common fixed-context discrete
evaluation, independent of the learned embedding. Its score is shared across
arms when the slot-labelled graph and prepared context are identical. An
improved continuous objective with an unchanged or invalid hard graph is not
evidence of improved molecular design. Report molecules and starts, in addition
to correlated checkpoints and directed atom/pair counts.

## Fixed-context hard scores and fresh native confirmation

The implemented hard score adapts `hard.to_relaxed(...)` through `adapt_graph`
and the same frozen scorer at the exact held pose, without learned embedding
injection. The existing `FrozenBoltzDBMol.discrete_reference` helper supplies
the native integer trunk/head reference used in interface validation; the
comparison's hard audit does not call that helper. Both routes inherit the
prepared atom names, reference positions, crop, masks and other context. Label
the implemented result **fixed-context hard projection**, not independent
native re-featurization of a changed molecule.

The existing `fresh_trajectory.execute_request` offers a separate ordinary
native confirmation route. It writes a fresh input YAML from each candidate's
canonical SMILES, preserving the declared protein/MSA; calls the pinned Boltz
CLI with predeclared seeds/settings into a new directory; then executes
`processed_identity.audit_processed`. That audit compares heavy-atom identity,
per-atom charges and typed bonds, complete ligand coverage, native generated
atom names/features, target and MSA linkage, and the affinity artifact's record
association. Native standardization that changes the molecule is an ineligible
result, with its reason retained.

This native CLI route generates a **new pose**, reference conformer and
chemistry-dependent native fields. It is a fresh-pose affinity follow-up, not
the same fixed-pose full DBMol objective and not a derivative through diffusion.
Its native affinity schedule can differ from the optimization schedule only
when that distinction is declared and applied equally to all candidates.
Fresh structure/confidence predictions alone do not reconstruct the selected
fixed-pocket contact objective.

Freeze candidate selection before inspecting any native outcome. Include each
paired original start, both terminal states and both best-own-objective states:
**five memberships per start, 15 total**. Deduplicate identical strict-valid
canonical molecules while retaining every arm/start/state membership. Each
unique valid identity receives pose seeds **17 and 23**, for at most **30 native
requests**. Invalid states retain their failure reasons and receive no disguised
replacement or repair. This initial requested comparison has no random-control
arm; conclusions concern the two declared surfaces and their shared starts.

The native schedule is 100 structure and affinity sampling steps, one diffusion
sample each, zero structure recycles, native affinity defaults of five recycles
and `bf16-mixed`, 128 MSA rows, maximum MSA parse length 8192, and no molecular
weight correction. These are applied equally to every request. Save all
preprocessing, prediction, identity and timeout failures, hashes, costs and
incomplete pairs. Repeated identities reuse the same native request rather
than receiving extra favorable random draws.

The authenticated native template is
`/work3/s234818/latentmol/boltz-preparation/source054-active-set-attempt001/count-13/input/carbon_chain_13.yaml`,
SHA256 `ff3f32119c9f6a51b37169306de81e43e60327797e79ae7a53ef90ee5a445117`.
Its MSA is `seq1.a3m` in the same input directory, SHA256
`d98f3b09f429e4fb087d67ca3b89f00c3165553444ef221ead6ed2cc01488bd4`.
The protein sequence SHA256 is
`978a204bab1837aa3304a838a621e6ef24fb4e878395db76c9c9ed095ce25f48`.
A [read-only standard-library SSH check](../outputs/implementation-checks/embedding-comparison-native-source-readonly-20261023.json)
confirmed those available bytes while
preparing this comparison; no model or native preprocessing ran on the login
node. The original C13 ligand entry is replaced by each exact candidate SMILES
before ordinary native parsing.

The implicit native affinity recycle/precision/MW settings were checked against
the installed `boltz/main.py`, SHA256
`b5ded4dac1552af2e079c335d56e64ea580b4d5ca0202a9542385ce3e318c3b3`:
lines 1372–1375 set five affinity recycles; line 1262 selects `bf16-mixed` for
Boltz2; and the affinity MW flag defaults false and is passed into model loading.
The new follow-up checks these source bytes and records their provenance before
requests, so an unexposed default is bound to actual native code.

`embedding_comparison_rescore.freeze_selection` freezes the complete role
denominator and deduplicated request list. `load_target` checks the template
against the prepared target/MSA identities. `run_followup` executes serial
ordinary native requests only inside a sealed allocation and records partial
execution if insufficient shared time remains. An internally corrupt native
identity audit stops the stage; an ordinary native prediction failure remains
a terminal failed request. A paired gain is eligible only when start and
candidate both pass the actual processed-input identity audit at the same seed.

Success in this small study could establish a useful local optimization result
and whether it survives ordinary native rebuilding. It cannot establish general
bridge accuracy, chemical synthesizability, experimental affinity or superiority
across targets and latent starts.

## Execution and preservation

All numerical tests, native preprocessing, derivative evaluations, optimization
and native predictions run inside allocated DTU HPC jobs. Local preparation is
limited to source edits and standard-library syntax/byte/provenance work. New
immutable attempts retain exact source, manifest, plan, model, bridge, prepared
context, start tensors and result hashes. A queued allocation is not a passed
gate or an optimization start; partial runtime results are not completed paired
budgets.

Use serial bounded stages with no allocation beyond 24 hours and no internal
budget beyond 23 hours, preserving fresh quota checks and remaining-time
headroom before child processes. No historical sealed source, failed decision,
mapping-pilot record or independent source136 continuation artifact is rewritten.
