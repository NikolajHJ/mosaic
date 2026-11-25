# Epoch-36 continuation collapse: mechanism work and independent weak test

Status, 21 October 2026 11:19 DTU-local: **both source132 quarter-step proof arms
FAILED early health at 200/2,500 updates after passing native readiness**.
The independent weak test ran despite baseline failure; neither arm qualified.
Training was not submitted, and epoch 36 remains the latest completed checkpoint.
An explicit quarter-AdamW-displacement candidate is proposed, not a confirmed
root-cause correction or validation pass. All model/chemistry diagnostics, numerical fixtures,
updates and evaluations remain on allocated DTU HPC. Local editing, static
review, packaging and saved-metadata inspection are permitted; this document
does not itself execute a test or change a sealed run.

## Both quarter-step proof arms failed at update 200

The [11:18 DTU-local status snapshot](../outputs/hpc-results/source132-epoch36-collapse-step-self-attempt001/status-status014.json)
records baseline GPU **29769067 EXIT**, its callback **29769068 DONE**, weak-test
GPU **29769420 EXIT**, and after-weak callback **29769421 EXIT**. The baseline
callback correctly submitted the independent weak test after baseline terminal
authentication. The final callback stopped because the baseline did not pass;
no training segment was submitted. Each arm passed native readiness and **187
mandatory CUDA fixtures** (no failures/errors/skips), then failed the same two
early-health guards at absolute update **237656**: raw-prior validity retention
and strict-teacher validity retention against the epoch-36 baseline. The other
16 health checks passed; this is a real scientific health failure, not the
earlier initial readiness placeholder.

[Saved fixed-panel outcomes](../outputs/hpc-results/source132-epoch36-collapse-step-self-attempt001/fixed-panel-outcomes-20261021-001.json)
give the matched 1,024-prior counts below. Raw chemical validity and strict
teacher representation acceptance are separate measurements.

| Disposable updates | Baseline raw / strict valid | Weak alpha-0.01 raw / strict valid |
| --- | --- | --- |
| 0 | 641 / 641 | 641 / 641 |
| 32 | 633 / 633 | 643 / 643 |
| 100 | 623 / 623 | 636 / 636 |
| 200, failed | 587 / 586 | 616 / 616 |

At update 200, raw validity was **57.3242% baseline / 60.1563% weak**, down
**5.2734 / 2.4414 percentage points** from 62.5977%; both exceeded the unchanged
two-point limit. Exact and chemically valid held-out reconstruction remained
**256/256 in both arms** at every saved panel. The baseline/weak failure-gate
SHA256 values are `16b8412edc75cee9f00abc98479d1ce84621702d642ce4e0ff7842d07c07e3de`
and `f211d3b577e39d311bd391b3055dba1400485206be877287a25dd1cd92bd2b8b`.

The same saved artifact retains **unfiltered categorical generation**, four draws
for each of the same 1,024 latents, without repair. Valid draws were initially
**2,091/4,096**, versus **1,968/4,096 baseline** and **2,000/4,096 weak** at the
failed 200-update endpoint. Unique valid identities were **1,976 → 1,859 / 1,888**.
Thus weak self guidance retained more validity than the matched baseline here,
but both worsened from their common start and failed health. This single
matched disposable comparison does not establish a successful correction,
replicated benefit, or a 2,500-update/three-epoch result. Draws sharing a latent
are correlated; they are not 4,096 independent latent examples. Failed states
remain nonpromotable diagnostics; no fix, waiver, resubmission or restart was
made during this saved-status-only update.

## Mechanism completed and restored; earlier candidate proof running observation

The [08:16:23 UTC mechanism snapshot](../outputs/hpc-results/source132-epoch36-collapse-step-self-attempt001/status-spike011.json)
records GPU **29768917 DONE** and callback **29768918 DONE**. All **100/100**
historical teacher receipts matched, the fixed raw-valid panels reproduced
**641 → 674 → 357/1,024**, and all eight snapshot restoration components passed.
Run `collapse_diagnostic/20261021T080306Z_chemistry-collapse-mechanism_6c1a6754`
took **758.556412 seconds**; result SHA256 is
`0591a673821bdf177faa39ab9fd319060deca3664f2d3b4f3564f6d584a10283`.

At update **96**, the before/full/quarter raw-valid counts were **590/531/580**.
The true frozen-target total losses were **0.06584161/0.06178990/0.06086013**:
the full step still lowered total loss. Weighted reconstruction changed
**0.00716297 → 0.00150318 full / 0.00200215 quarter**, while weighted prior
repair rose **0.00204124 → 0.00327266 full / 0.00214578 quarter**. Decoder
reconstruction gradient norm was **0.2247771**, versus prior repair **0.0080407**
(about 28 times); prior-repair gradient dot the actual full decoder displacement
was **+0.0001089184**. Full/quarter bond-label flips were **1,645/361**, and atom
flips **442/115**. Together with update 39, this establishes a local shared-decoder
objective tradeoff, not a sole explanation of the collapse. The quarter proposal
at update 96 starts from the replay's already-changed full-step trajectory; it
is not a 100-update quarter-step result. Lower local harm does not establish
longer-term health or a validated correction.

The [08:18 UTC validation snapshot](../outputs/hpc-results/source132-epoch36-collapse-step-self-attempt001/status-validation012.json)
supersedes the preceding queued observation: baseline GPU **29769067 RUN**,
started about **10:17 DTU-local**, with one A100/16 CPU slots/24 hours; callback
**29769068 PEND**. Fresh native checks and the **2,500 disposable updates** remain
pending. The independent alpha-0.01 weak arm uses the same quarter-step policy
and original epoch-36 start; it is scheduled only after baseline terminal
authentication, regardless of baseline scientific success, and is not yet queued.
The [08:21:10 UTC snapshot](../outputs/hpc-results/source132-epoch36-collapse-step-self-attempt001/status-validation013.json)
adds **187 guarded CUDA fixture passes**, no failures/errors/skips, in **33.856
seconds** (JUnit SHA256 `0d5a24e9a87fc4070c820a447ddc9ce35c500b1ca22b2328124f483d85628127`).
The native benchmark remains active; the initial `failure_readiness.json` failed
placeholder has no gates/error and is not a terminal scientific result.
Training is not queued or started. All original gates remain mandatory, and any
eligible continuation must discard proof state and restore original epoch 36.

## Partial update-39 finding; replay and proof incomplete

The [08:09:47 UTC spike snapshot](../outputs/hpc-results/source132-epoch36-collapse-step-self-attempt001/status-spike009.json)
has replay **49/100**, with historical teacher receipts matching so far and
fixed panels 641 raw-valid initially/674 at update 32. Update 96, the full
100-update replay and exact final state/RNG restoration remain pending.

At update 39, the **true frozen-target full scalar decreased** from
**0.06859208647 to 0.06129076865** (about -0.0073013). Its weighted
reconstruction term fell **0.0120312 → 0.00421343**, while weighted prior
repair **rose 0.00169576 → 0.00218443**. Decoder reconstruction gradient norm
was **0.20447045**, versus prior repair **0.00764702** (about 26.7 times).
The actual decoder displacement had reconstruction gradient dot delta
**-0.02159166**, but prior-repair dot delta **+7.7046254e-5**. This is
update-specific objective interference/tradeoff; it is **not an observed
increase of the total finite-step objective or proof of total-loss overshoot**.

The same fixed-prior observation had **637 raw-valid before update 39**, then
**625 at full displacement** and **633 at quarter displacement**. Reported
argmax bond-label flips were **1,033 versus 249**, atom-label flips **336
versus 89**, charge flips zero. Quarter displacement mitigates this single
update's change; it does not establish the 100-update mechanism, a successful
correction or longer stability. Await update 96, the complete replay/restore
checks and all candidate 2,500-update/health gates. No fix or training is
adopted from this partial result.

## Full CPU passed; mechanism running with CUDA fixtures passed

The [08:03:09 UTC receipt/status snapshot](../outputs/hpc-results/source132-epoch36-collapse-step-self-attempt001/status-mechanism006.json)
confirms CPU **29768881 DONE**, after-CPU **29768882 DONE**, and saved full
CPU gate/repair profile passed. Mechanism GPU **29768917 RUN**, started
approximately **10:02 DTU-local** on an A100 with 16 CPU slots and a **two-hour
allocation**; callback **29768918 PEND**. Its mandatory CUDA JUnit records
**60 passed, zero skips/failures/errors**, **13.39s**, SHA256
`21df174844b098ef8adf88556d2b2b9edb8e85ec2c4bb07e6451d1a62d61f188`.
These fixture passes are not a completed mechanism replay or a causal/fix
result. Candidate baseline/weak native and 2,500-update proofs are **not yet
submitted**, and training is not submitted. All actual stage receipts remain
bound to the sealed source132 identities below.

## Allocated units passed; full CPU stage not complete

The [08:00:42 UTC snapshot](../outputs/hpc-results/source132-epoch36-collapse-step-self-attempt001/status-cpu004.json)
records **1,732 collected: 1,648 passed, 84 CUDA-only skips, zero
failures/errors**, JUnit **226.191s**, SHA256
`2475713ab573f86b32018664ab97abcfbcb6180adbe7863547ffaf12b4747de8`.
CPU **29768881 remains RUN**, finishing the original repair profile. No
`cpu_gate` is saved yet, so the full CPU stage is **not passed/completed**.
The mechanism GPU is not queued; no native/proof result or training exists.
Unit success does not substitute for the profile or later scientific gates.

## Source132 actual immutable staging and CPU submission

[Preparation](../outputs/hpc-submissions/source132-epoch36-collapse-step-self-attempt001-preparation.json)
records the unique source132 release, candidate plan and independent weak-test
plan. The [stage-and-queue receipt](../outputs/hpc-submissions/source132-epoch36-collapse-step-self-attempt001-stage-and-queue-001-output.json)
returned 0, authenticated **612 members**, and submitted CPU **29768881** and
after-CPU **29768882** as `submitted_not_passed`. The
[07:57:33 UTC snapshot](../outputs/hpc-results/source132-epoch36-collapse-step-self-attempt001/status-cpu002.json)
shows CPU **29768881 RUN** on 16 slots and after-CPU **29768882 PEND**.
No completed CPU pass or GPU mechanism/native/stability/training submission
is established. Preparation's `sealed_not_submitted` field is historical;
the separate operation receipt records the later submission.

| Source132 identity | SHA256 |
| --- | --- |
| Archive | `7fa8447292faa99356ecac414f933622bf7c8f409687fb5176b92f45ca165552` |
| Manifest | `faa8ae814075a32aa73215948b3c14754deca8ecfeaa68bbb61850afce7f2362` |
| Candidate plan | `0eaa153fc39caa68ef83667b078d13824f991944680be3f7c7ec4e4b53507695` |
| Independent weak-test plan | `2d334ee7be1656b4dcf172f93f0fca1a2127ec37c068759f8985590d4514c099` |

The [candidate plan](../outputs/hpc-submissions/source132-epoch36-collapse-step-self-attempt001.json)
and [independent weak plan](../outputs/hpc-submissions/source132-epoch36-collapse-step-self-attempt001-self-test.json)
retain original epoch-36 lineage, loss/ramp and the matched quarter-displacement
amendment. Future GPU allocations are serial: mechanism diagnostic at most
**two hours**, then each baseline/weak proof at most **24h** with the declared
internal budget. None is queued yet, and the candidate is not a proven fix.

Root's operator command tails, through bundled CPython
`C:/Users/nikol/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`, were:

```text
outputs/hpc-submissions/prepare-source132-epoch36-collapse-step-self-attempt001.py
outputs/hpc-submissions/operate-source132-epoch36-collapse-step-self-attempt001.py stage-and-queue
```

This is local static/byte packaging, transfer and scheduler work only, not
local numerical validation. Source131 and earlier failures remain unchanged;
only workspace documentation is updated, never the sealed archive.

<a id="source132-frozen-design-awaiting-immutable-sealing"></a>

## Source132 design frozen before sealing

The planned identity is `source132-epoch36-collapse-step-self-attempt001`.
The retrieved [source129 mechanism evidence](../outputs/hpc-results/source129-chemistry-validity-runtime-diagnostic-attempt001/status-mechanism20261021-001.json)
reports a **negative full-gradient dot actual AdamW displacement**, so an
uphill full-objective step is not established. Its quarter-scale one-step
observation retained **641/1,024 raw-valid**, versus **625/1,024** at full
scale; an eight-update prefix still failed retention. Equal counts do not
mean identical graphs, and one-step retention does not establish stability.
The [source131 collapse artifacts](../outputs/hpc-results/source131-short-callback-result-attempt001/collapse-mechanism-artifacts-20261021-001.json)
and original 0/32/100 counts below remain authoritative. No single mechanism
or successful fix follows from these observations.

The first stage must exactly replay source131's **original 100-update stream**
from epoch 36, with original weights/moments/RNG/teacher and no step amendment.
Instrument the selected spike updates **39 and 96** with weighted component
gradients and a **true frozen-target scalar** at the full and quarter versions
of the same inherited-AdamW proposal. Hold that state's real/prior inputs,
latent draws, masks and detached targets fixed throughout the comparison.
Use the actual aggregated **nonlinear population JS**, not the streamed
cotangent surrogate as a reported finite-step objective. Component values,
gradients, first-order dots and finite-step scalar changes remain distinct.
Modules `chemistry_collapse_diagnostic` and `chemistry_optimizer_step`, plus
a new suite controller, implement this prepared scope; replay fidelity and
all numerical claims still require allocated checks.

The candidate declares **factor 0.25 on the actual AdamW parameter displacement**:
`theta_applied = theta_before + 0.25 * (theta_native_AdamW - theta_before)`.
Keep the usual inherited moment/counter update and original objective,
reference 35/ramp, search, ratio and other settings. This is a versioned
**update-size amendment**, mathematically a reduced parameter step; it is
**not gradient clipping**, a gradient-sign correction or a proved root cause.
Retain the original clipping policy separately. Native proposal/delta
formation, moment preservation and actual applied tensors need explicit tests.

The independent alpha-0.01 weak-self test uses the **same factor 0.25** from
the same original epoch 36, even if the baseline candidate fails. It is still
disposable/nonpromotable, with no automatic weak adoption. Both candidates
retain fresh native readiness, all early 32/every-100 health gates and the
complete 2,500-update requirement; only a fully passing corrected candidate
can restore epoch 36 for conditional serial training ending at epoch 39.

The [saved static review](../outputs/hpc-submissions/source132-static-review-20261021-001.json)
records bundled-Python **AST and whitespace checks for 23 Python files**:
standard-library parsing/source inspection only, with no project imports,
numerical tests or model evaluation. At that pre-seal stage no immutable seal,
job or allocated result existed yet. Later identities/commands are recorded
above; allocated validation is still pending.

## Evidence to explain, not overwrite

Source131's [saved outcome](../outputs/hpc-results/source131-short-callback-result-attempt001/fixed-panel-outcome-news20261021-001.json)
shows the original source121 loss passed native readiness, including 85 CUDA
fixtures and all 18 native/11 supplementary gates, then failed the longer
proof at **100/2,500** disposable updates. The matched 1,024-prior raw-valid
counts were **641 → 674 → 357** at 0/32/100 updates; strict-teacher counts
were **641 → 674 → 354**. The two starting-validity retention checks failed;
the other 16 early-health checks passed. Held-out exact/valid reconstruction
remained 256/256. At 100, 667 failures were chemical and three representation,
with zero input failures. These are measured outcomes, not a causal diagnosis.

The weak-self stage did not run, so neither its benefit nor harm is established.
The original-loss result means the later failure-guidance terms alone cannot
explain every observed collapse. It does not prove an AdamW, implementation,
bond, atom, sulfur or data mechanism. Preserve source127/source128/source130/
source131 failures, source129's sealed diagnostic and the original epoch-36
checkpoint; do not rewrite the failed controller decision to run an old arm.

## Targeted mechanism evidence

Authenticate the retrieved source129 probe and source131 100-update logs/
diagnostics to their source, plan, parent and run identities before analysis.
Use saved metadata where sufficient; any re-evaluation of a failed state or
new matched update must be a separately recorded allocated diagnostic. A
failed-state snapshot is nonpromotable and cannot replace epoch 36.

Compare fixed-coordinate graph transitions and separate chemical, serialization,
schema and canonical-roundtrip failures. Inspect actual bounded atom/charge/
unordered-bond label changes, categorical marginals and target provenance.
Preserve attempts, failed/not-evaluated stage counts and raw-versus-strict
validity; do not infer a population mechanism from changing teacher batches
or a reconstruction scalar alone.

Use the smallest necessary matched diagnostics to distinguish objective
contributions, current branch gradients, actual inherited-AdamW displacement
and finite-step panel effects. The verified decomposition permits
`g_real·g_prior = (||g_full||²-||g_real||²-||g_prior||²)/2`, also on the decoder.
Compare it with `g_prior·actual_delta` and encoder/decoder displacement summaries.
These test branch conflict versus moment/update alignment, without turning a
first-order sign into proof of chemical validity or causal attribution.
Declare any additional diagnostic comparison before running it; avoid a broad
parameter grid in place of mechanism evidence.

Choose only a correction supported by that evidence. If optimizer-step or loss
semantics change, declare their exact version/configuration and preserve parent
AdamW moments and RNG rather than quietly resetting them. Execution-only
optimizations require target/loss/gradient/classification parity. No particular
correction is selected or assumed successful by this authorization.

## Independent weak-self experiment

The user now authorizes the weak arm **independently of whether the baseline
passes**. Give it a fresh immutable plan/run/controller identity and its own
readiness/health checks. This changes the future dependency policy, not
source131's recorded blocked weak stage. It does not waive the weak arm's own
gates or permit it to train/promote automatically.

Restore original epoch 36 and use the declared strict-valid prior self-target
rule. The term is `+0.01 * mean_over_all_prior(normalized_full_self_NLL)`,
with zeros for other rows, independent of the 0.25/0.75 repair weights/ramp.
Full normalization is `N + K + K*(K-1)/2`; targets/masks are detached and
authenticated, and self rows do not create repair-witness counts. Keep the
comparison matched to the baseline's initial state, teacher/input/RNG scope
and update budget; do not start from the failed 100-update state or another
arm's trained endpoint.

Run applicable native readiness followed by up to the declared **2,500
disposable updates**, stopping honestly on failed health. Retain panels at
0/32/every 100 and before/after unfiltered generation on fixed 1,024 latents
with four categorical draws per latent, alongside raw argmax outcomes.
Report sample/argmax validity, strict representation acceptance, diversity/
unique-valid counts, reconstruction, composition/sulfur and runtime with
all-attempt denominators. Retain invalid graphs and failed/partial evidence.
Self-NLL improvement is not a validity/diversity guarantee. No weak result
automatically selects a training objective.

## Unchanged handoff and continuation gates

All candidates/test arms authenticate original epoch 36, checkpoint SHA256
`3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f`,
its AdamW/native and fresh-prior RNG, frozen reference 35/original ramp,
1:2 real/prior ratio, search and other scientific settings. Preserve 256 real/
512 prior full updates, microbatch 128, 12 workers/four main Torch threads
within 16 CPU slots, serial 24h allocations and 23h internal runtime.

Fresh CPU/profile, native/parity/finite-value, chemical/representation retention,
reconstruction, composition/sulfur, memory, runtime, resume and quota gates
remain mandatory. Keep the full 2,500-update proof and fixed-panel schedule;
no failed short/native/early gate can be relabelled passed. One-time startup
and shared-clock accounting do not remove slow recurring samples, overhead
or conservative fit guards. Preserve and explain explicitly inapplicable
fallback-only assertions without weakening general health requirements.

Only a fully passing corrected candidate can proceed to baseline continuation.
Discard all disposable model/optimizer/RNG changes, restore original epoch 36,
authenticate no competing continuation and fresh quota, then submit serial
training for **absolute epochs 37–39 only**. The global maximum remains 100,
but no automatic epoch 40 is authorized here. Weak tests remain nonpromotable.
Actual source/configuration hashes, commands, allocations, receipts and every
outcome will be recorded in the ledger when available.
