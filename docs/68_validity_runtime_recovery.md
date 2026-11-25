# Targeted recovery of chemistry validity and execution time

Current continuation scope is now the separately sealed
[source130 original-loss baseline and weak-self test](69_short_baseline_and_valid_self_guidance.md),
with source130 CPU validation failed and a separate sealed source131 interface
correction whose full CPU/native readiness passed but whose original-loss
2,500-update proof failed fixed-panel retention at update 100; no weak-test
GPU or training started. The source129
diagnostic and earlier unsealed-preparation observations below remain history;
no source129 mechanism result or selected validity fix is inferred.

Status, 20 October 2026 19:08 DTU-local: **source129 CPU stage passed; GPU probe PEND**.
No optimizer-step control has been selected or validated. There is no GPU test
result yet, and training is not submitted.
All numerical
diagnostics, tests and evaluations remain allocated DTU HPC work. This document
records requirements, not observed mechanism or successful recovery.

## Diagnostic still waiting; no GPU results, 20 October 2026 19:08 DTU-local

The [17:08:05.896676 UTC snapshot](../outputs/hpc-results/source129-chemistry-validity-runtime-diagnostic-attempt001/status-news001.json)
confirms diagnostic GPU **29763843 PEND**, with no GPU result files. The CPU
job/callback have now aged out of live LSF records; their earlier DONE and
passed CPU gate/profile evidence remains authoritative. The combined `bjobs`
return code 255 concerns those missing old records and is **not a GPU failure**.
The diagnostic has not finished. Next are the allocated mandatory CUDA
fixtures and mechanism probe; only after those saved results are authenticated
can a minimal validity correction be selected for a new immutable candidate
and fresh native/2,500-update checks. No code change, source130 seal/submission
or training occurred in this read-only status turn. Unsealed preparation and
all prior failures remain as recorded below.

## Current blocker and next evidence, 20 October 2026 18:16 DTU-local

The [16:16:01 UTC snapshot](../outputs/hpc-results/source129-chemistry-validity-runtime-diagnostic-attempt001/status-gpu004.json)
confirms CPU/callback DONE and diagnostic **29763843 PEND**. No GPU fixtures,
native mechanism-probe result or training is established. The immediate blocker
is the A100 allocation; the validity mechanism remains unconfirmed and no
validity correction or optimizer-step control is selected.

Once the authenticated diagnostic exists, inspect the saved weighted full,
real and prior gradient norms. Since the capture verifies
`g_full = g_real + g_prior`, compute
`g_real · g_prior = (||g_full||² - ||g_real||² - ||g_prior||²)/2`, also for the
decoder subset. Compare branch alignment with the separately saved
`g_prior · actual_AdamW_delta` and decoder displacement diagnostics. These
distinguish current-objective branch conflict from inherited-moment update
alignment; they are first-order observations, not proof of finite-step loss,
validity or the historical failure's cause. **No derived result exists yet.**

The next sequence is explicit: await/verify the source129 GPU evidence and
paired chemical-versus-representation transitions; choose the minimal justified
validity correction; seal a new candidate with declared startup/parity/clock
policies; pass fresh CPU/native and all 2,500 disposable updates/early panels;
then discard test model/AdamW/RNG changes and restore original epoch 36 before
any conditional, serial training submission with fresh quota. Diagnostic
success alone cannot skip a stage or promote a state.

## Unsealed startup and runtime-accounting preparation

Static source review confirms a distinct issue in the old longer-proof path:
a newly created cold worker pool could put one-time process/import cost into
the first timed update, then extrapolate that maximum across the whole epoch.
This is a code-path/accounting concern, not an observed explanation for
source128's already-warmed native runtime rejection; source128 never reached
that longer proof. It does not establish a validity mechanism or a speed gain.

The unsealed opt-in `worker_startup.version = "chemistry-worker-startup-v1"`
prepares the existing cold pool through its public `start()` hook. Native
single-thread initialization precedes imports of the repair/progress/classifier
modules; a 12-worker barrier prevents a few fast workers from substituting for
all initializers. Warmup classifies 256 authenticated real training graphs,
without decoder forwards, random latent draws, repair searches, optimizer
steps or target-cache insertion. A nonempty cache or already-started pool is
rejected. Model/reference, AdamW, named gradients, modes and all scientific RNG
streams are restored exactly; failures retain evidence and close owned workers.

The shared monotonic clock starts before the native benchmark and is carried
into stability setup, startup and updates without resetting. The warmed epoch
projection retains **every measured update**, uses the **slowest update** and
**maximum observed panel time**, charges startup once in that projection, and
retains the **600s base overhead**. It takes the maximum of native and warmed
conservative estimates, then applies the unchanged outer fit guard. Metadata
authentication binds the warm receipt to its exact run/job/source/plan/data/
checkpoint, zero-update state-restoration proof and input identities, and
recomputes the projection from saved update/panel/startup walls. No slow sample,
old gate, reserve or deadline is removed.

Actual startup also remains in the elapsed allocation clock. The training
caller's fit check conservatively adds elapsed setup to an estimate containing
startup; this safety headroom is intentionally retained, **not exact
nonduplicated total accounting**. Neither the existing epoch-fit gate nor the
elapsed-allocation check is loosened.

The [ten-file static receipt](../outputs/hpc-submissions/unsealed-worker-startup-static001.json)
pins the reviewed AST/source hashes: standard-library syntax/hash checks only,
without project imports or numerical tests. Startup/shared-budget/authentication
fixtures are prepared, not passed. **No source130 seal, submission or training
exists**; sealed source129 remains unchanged while its probe waits.

## Latest wait and separate preparation, 20 October 2026 18:05 DTU-local

The [16:05:21 UTC snapshot](../outputs/hpc-results/source129-chemistry-validity-runtime-diagnostic-attempt001/status-gpu002.json)
still records GPU diagnostic **29763843 PEND**, with no GPU test/probe result
or training. A separate [narrow cache cleanup](50_hpc_space_cleanup.md#narrow-generated-cache-cleanup-completed-20-october-2026-1559-utc)
removed 6,212 generated cache files with verified durable receipts, preserving
active/referenced source and scientific artifacts. Official quota after cleanup
was 10.76 GiB free; this is not a reservation or promotion waiver.

Optional startup hooks, a shared native/stability clock and runtime helper/
fixtures are being prepared separately. No source130 is sealed/submitted and
no optimizer-step policy is selected. These unsealed edits do not alter the
existing source129 diagnostic. The earlier cold-start concern remains separate
from source128's already-warmed native runtime failure.

## Actual bounded GPU diagnostic submitted

The [15:57:36 UTC snapshot](../outputs/hpc-results/source129-chemistry-validity-runtime-diagnostic-attempt001/status-gpu001.json)
confirms CPU **29763813 DONE** and after-CPU **29763814 DONE**. The latter
submitted GPU probe **29763843** at **17:56:01 DTU-local**: one 80GB A100,
16 CPU slots, **60-minute requested walltime**, diagnostic only. It is **PEND**;
16 slots were reserved on `n-62-18-9` at 17:56:04, with an affinity-availability
wait across two hosts. This is a scheduler resource wait, not a GPU test or
mechanism result. No post-probe continuation callback or training is submitted.
The source/plan and bounded diagnostic policy below remain unchanged.

Separately, root/runtime review identified cold-start overprojection in the
later long-proof runtime-estimation path and is preparing an **unsealed**
helper correction. That separate finding does not explain source128's already-
warmed native runtime failure. It changes no sealed source129 file or outcome
and is not a validated fix or adoption of a continuation candidate.

## CPU gate passed; scheduler completion not yet observed

The [15:55:59 UTC snapshot](../outputs/hpc-results/source129-chemistry-validity-runtime-diagnostic-attempt001/status-cpu002.json)
retains a passed CPU gate and original repair profile, with **1,395 collected:
1,325 passed, 70 CUDA-only skips, zero failures/errors**, JUnit **218.091s**.
The CPU 29763813 wrapper logged `Successfully completed` at 17:55:47 DTU-local,
and its saved allocation `run.json` is completed. However, the live LSF query
12 seconds later still reports **RUN**, while after-CPU **29763814 is PEND**.
Keep the saved scientific gate/allocated result distinct from that scheduler
observation; do not report either callback completion or GPU submission yet.
**No GPU diagnostic or training is queued.** Wait for authenticated actual
callback/probe receipts, rather than treating a passed CPU gate as submission.
Source129 remains diagnostic only, and source127/source128 outcomes are intact.

## Source129 actual staging and CPU submission

[Preparation](../outputs/hpc-submissions/source129-chemistry-validity-runtime-diagnostic-attempt001-preparation.json)
binds 16 exact overlays, including actual uncommitted code/tests, to the new
diagnostic release. The [stage-and-queue receipt](../outputs/hpc-submissions/source129-chemistry-validity-runtime-diagnostic-attempt001-stage-and-queue-001-output.json)
records return code 0, **588 members authenticated**, and actual CPU
**29763813**/after-CPU **29763814** submitted as `submitted_not_passed`.
**GPU diagnostics are not yet queued; training is not submitted.** The earlier
preparation field `sealed_not_submitted` remains its original historical state;
the separate queue receipt records the later action.

| Identity | SHA256 |
| --- | --- |
| Source archive | `1b08567278f59b0c47b8c394e5da09495a243b4a5b850eed5d8f2d89323be98d` |
| Source manifest | `2dc4025e6251076c6e57541f0428e62e9461db3face1ab35fcf08e26b0200e67` |
| Immutable plan | `d8e29d5488075bbe7b483796a1f88fa533b31e1200321498f76f42ed0b293f0a` |

Root's local operator commands use bundled CPython at
`C:/Users/nikol/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`:

```text
outputs/hpc-submissions/prepare-source129-recovery-diagnostic-attempt001.py
outputs/hpc-submissions/operate-source129-chemistry-validity-runtime-diagnostic-attempt001.py stage-and-queue
```

These are standard-library syntax/packaging/byte staging and scheduler work,
not local numerical tests. The plan explicitly declares execution-only
`classification_execution.version = "repair-classification-pool-v1"` and
`recovery_diagnostic.version = "epoch36-optimizer-mechanism-v1"`. Loss, parent
36/reference 35, 1:2 sampling, original ramp and other scientific settings are
unchanged; execution equivalence still requires allocated parity tests. This
release has **not chosen a validity fix or optimizer-step policy**. All source127
stopped/source128 failed records remain preserved.

<a id="source129-diagnostic-only-stage-under-preparation"></a>

## Source129 diagnostic-only controller and scope

Source129 is **diagnostic only**, not the new training
candidate. Its controller is
[`chemistry_recovery_diagnostic.py`](../scripts/hpc/chemistry_recovery_diagnostic.py),
with contracts in
[`test_chemistry_recovery_controller.py`](../tests/test_chemistry_recovery_controller.py).
Read-only source review found the following intended boundaries; these are
static observations, not executed fixture or model results:

- The CPU command retains all original guarded-restart contracts/profile and
  appends classification-pool, matched-classification-validation, mechanism-probe
  and diagnostic-controller tests.
  `GuardedRestart.cpu_contract_command()` is an overridable instance method;
  its default still returns the original command, and gate authentication uses
  the actual instance command rather than a shorter substitute.
- Only `queue`, `cpu` and `after-cpu` are CLI modes. Constructing other callback
  modes or extra mode arguments is rejected, and `advance()` always raises.
  After CPU success, stopped-predecessor authentication and old-oracle SHA
  authentication, the controller may queue **one probe using an 80GB A100,
  16 CPU slots and at most one hour**.
  Mandatory CUDA fixtures precede the probe with shell `&&`, so fixture failure
  cannot proceed to model diagnostics.
- The GPU probe has **no post-probe continuation callback**. It cannot launch
  the 2,500-update stability proof or training. Its updates/states are
  disposable; probe success is diagnostic evidence only, never readiness or
  adoption of an optimizer-step policy.
- Inherited plan initialization and the allocation wrapper still authenticate
  plan/archive/manifest/parent and release bytes. The exclusive after-CPU
  intent prevents duplicate probe submission after a lost reply; an existing
  decision must match the current identity. Failed CPU/predecessor/reference
  checks stop before queuing the probe and retain a stopped decision.

The reviewed controller tests cover forbidden promotion/advance modes, exact
extension of CPU contracts, no GPU queue after CPU failure and one bounded
probe after mocked successful guards. They now also cover replay identity,
idempotent re-entry, orphaned intent and predecessor-stop failure. No
CPU/GPU numerical pass or continuation is established by the seal/submission. The
later corrected continuation candidate must independently pass the complete
native/early-health/2,500-update/runtime/quota path below; the diagnostic is
not an alternate promotion route.

### Probe semantics and follow-up static review

The prepared probe observes one inherited-AdamW proposal on the first actual
production-shuffled batch, four declared displacement scales (1, 1/2, 1/4,
1/8), and separate eight-update ordered-benchmark and production-shuffle
prefixes. Snapshot/restoration covers model/reference parameters, named
gradients, module modes, optimizer state with independent moment storage,
Python/NumPy/Torch/CUDA RNG and named train/fresh-prior RNG. Each arm restores
the authenticated start, training shuffle and epoch-37 learning rate. Final
restoration is checked even on failure. These are disposable observations,
not a candidate state to resume.

Post-accumulate hooks observe the actual weighted real/prior full-objective
backwards, excluding diagnostic `autograd.grad` probes. Their sum is checked
against the complete pre-clipping gradient. Report whole-model and separate
encoder/decoder gradient-dot-displacement values, not isolated chemistry-term
gradients or finite-step descent certificates. A synthetic inherited-momentum
fixture tests a possible uphill displacement; it does not identify the cause
of source128's observed drop.

Two review findings were corrected before sealing: the GPU command now runs
the new diagnostic CUDA fixtures, and scale 1 loads the **exact saved AdamW
proposal state**, rather than reconstructing FP32 subtraction/addition. A
cancellation fixture covers that difference. Fractional displacement is formed
in float64; observing fractions does not step optimizer moments again. All
fixed-panel scale results are reported without selecting a winner or promoting
a policy. The first production-prefix update must reproduce the instrumented
proposal's model and ordered teacher receipt.

The 1,024-point observer now saves all row classifications at fixed identities,
paired validity-transition counts and at most four original before/after graph
examples per changed transition. Categorical label edits count unordered bonds
once. Separate original 256-prior baseline/prefix endpoints preserve the old
benchmark's denominator. Raw-validity transitions remain distinct from schema
roundtrip acceptance. No missing old graphs or probabilities are invented.

Teacher phase seconds are finite/nonnegative execution observations outside
scientific receipt identity. Full-update parity excludes only `search_seconds`,
`prepass_seconds` and `teacher_phase_seconds`; scientific metrics/counters,
receipt order, full gradients, model/AdamW/RNG and mode states still compare
exactly. Worker sums/coordinator phases can overlap and are not an exclusive
GPU breakdown. The root reported standard-library AST parsing of nine files
only, without imports or numerical execution. The matched-classifier helper
is included in the sealed release; all allocated tests/probe outcomes remain
pending.

## Why both checks remain required

Source128's [saved results](../outputs/hpc-results/source128-corrected-chemistry-guarded-restart-attempt001/result-summary-news001.json)
show 33 passing CUDA fixtures, passing native finite differences and nonzero
guidance gradients for 22/22 failed rows. Its original benchmark nevertheless
failed runtime: mean epoch projection 65,826.5391s requires 84,083.1739s under
the declared guard, above the unchanged 82,800s internal cap. Its fixed
256-prior panel also fell 149 to 142 raw-valid graphs after eight disposable
updates, a 2.734375-point drop beyond the supplementary 2-point limit. That
supplementary gate was not assembled after the original runtime failure.
Gradient existence/parity therefore does not establish a helpful update or
health. Exact chemical/optimization causes remain unproven.

The user authorized addressing **both** validity and runtime, rerunning the
checks from epoch 36, and conditional fresh continuation only after all pass.
Use a new immutable candidate. Never restart source127/source128 jobs, replace
their failure records or present a changed policy as their original pass.

## Targeted mechanism and execution work

Use the smallest necessary matched diagnostics rather than a broad parameter
grid. Authenticate epoch-36 model, AdamW moments, native/fresh-prior RNG and
epoch-37 learning-rate/shuffle lineage. Predeclare the comparisons needed to
separate objective-component behavior from the actual inherited AdamW update.
Distinguish component scalar values, component gradients, aggregate gradients,
optimizer displacement and matched-panel effects; one does not prove another.
Save authentic inputs/targets/classification and exact normalization/provenance
so any conclusion can be checked. Do not infer chemical or sulfur mechanisms
from unclassified changing training minibatches.

If an optimizer-step control is needed, declare its exact new version,
semantics, bounds and compatibility behavior in immutable configuration before
testing. Retain parent AdamW moments and RNG; do not silently reset the optimizer
or substitute a disposable test state. No particular step control is selected
or assumed to pass by this authorization.

Runtime work may remove duplicate classification, worker or receipt overhead
while preserving scientific semantics. Verify identical hard graphs, teacher
and classification stages/counters, search candidate order/budget/witnesses,
objective values and applicable gradients against the authenticated reference.
Do not turn omitted validation or changed failure handling into a speed claim.
Measure matched execution with the same state/inputs/resources; keep CPU timing,
asynchronous aggregate timings and end-to-end GPU update/epoch estimates distinct.

## Unchanged lineage, boundaries and staged acceptance

Retain the completed epoch-36 checkpoint SHA256
`3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f`,
its model/AdamW/native and fresh RNG, frozen reference 35 and original ramp.
Keep 256 real/512 prior (1:2), microbatch 128, 12 search workers plus four main
Torch threads in 16 CPU slots, other loss weights, composition/occupancy/KL and
bond-search semantics. The maximum remains 100 total epochs, serial allocations
of at most 24h and a 23h internal cap. The declared runtime factor 1.25/reserve
1,800s remains a real gate, not a waived runtime failure.

The [guarded restart protocol](67_guarded_failure_guidance_restart.md) remains
mandatory: allocated CPU contracts/profile, native CUDA/parity/finite-gradient
and short-update readiness, then the complete 2,500-update disposable proof
spanning the prior roughly 1,500–2,300-update deterioration window. Evaluate
fixed 1,024-prior/256-reconstruction panels at baseline, after 32 updates and
every 100 thereafter. Keep raw chemical validity and strict schema/roundtrip
acceptance separate, preserve bounded real failure examples/marginals, and
retain nonpromotable failed model/optimizer/RNG diagnostics.

Every existing native/source/input, finite-value, retention, sulfur/composition,
reconstruction, memory, runtime, resume and fresh-quota check remains required.
Include early-panel overhead in conservative fitting. Stop on a failed check,
exception, nonfinite result or exhausted budget; no partial proof is a pass.
Only after all gates pass may the controller restore original epoch 36
(discarding all disposable model/optimizer/RNG changes), authenticate no competing
continuation and fresh quota, and submit a new continuation. Authorization,
CPU success, native success, a queued job and completed training are distinct.

Exact implementation decisions, source/config hashes, commands, allocations,
receipts and outcomes will be added when available. Existing source127 stopped
data and source128's failed candidate remain authoritative historical evidence.
