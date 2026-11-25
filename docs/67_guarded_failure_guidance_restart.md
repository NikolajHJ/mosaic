# Guarded validation of failure-guidance scaling and continuation

The subsequent [validity/runtime recovery scope](68_validity_runtime_recovery.md)
is authorized preparation for a new immutable candidate. It retains this
document's staged/early-health requirements and source128's failed outcome;
no source128 restart or retroactive gate pass is authorized.

## Source128 native benchmark failed; no stability proof or restart, 20 October 2026 17:19 DTU-local

The [15:19:52 UTC status snapshot](../outputs/hpc-results/source128-corrected-chemistry-guarded-restart-attempt001/status-news001.json)
and [saved-result analysis](../outputs/hpc-results/source128-corrected-chemistry-guarded-restart-attempt001/result-summary-news001.json)
supersede the earlier queued observation. GPU **29760576** ran
09:25:56–09:31:12 DTU-local and its wrapper exited 1. Callback **29760577**
ran 09:31:15–09:31:17 and stopped at the failed native gate. Live scheduler
records have since aged out; persisted allocation/log/gate evidence establishes
the outcome. Source128's **2,500-update stability proof, classified early-health
baseline and training did not start**. Source127 stays stopped; completed epoch
36 remains protected.

The 33 allocated CUDA fixtures passed with zero failures/errors/skips in
16.817s. The native 1,024-prior panel's finite-difference check passed
(analytic `2.961395072e-7`, finite difference `2.961395066e-7`), and all
22/22 original failed rows had nonzero guidance gradient. These checks are
not a complete scientific readiness pass. The original benchmark passed
**17/18 gates**, failing `complete_epoch_fits`: projected mean epoch
**65,826.5391s (18.28515h)** gives `1.25*T + 1800 = 84,083.1739s`, exceeding
the 82,800s internal cap by **1,283.1739s (21m23s)**. Derived conservative
arithmetic from the slowest of eight updates gives **71,797.3736s (19.9437h)**,
requiring **91,546.717s (25.4296h)** under the declared guard. These are saved
runtime projections, not completed epochs; the original supplementary gate
was not assembled because the original benchmark raised first.

There is also a short-panel retention concern, so this must not be described
as runtime-only with all other health established. The fixed **256-prior
epoch-36 starting panel** was **149/256 raw-valid (58.203125%)**, falling to
**142/256 (55.46875%) after eight disposable updates**, a **2.734375 percentage
point drop**. That would fail the supplementary maximum 2-point drop, but
the supplementary gate was **not assembled**, rather than recorded as passed
or failed. Frozen reference 35's 4/256 is not the epoch-36 baseline. Held-out
exact reconstruction changed 256/256 to 255/256 (99.609375%); reconstruction
chemical validity remained 256/256 in both. This short result cannot establish
behavior through the earlier deterioration window or a specific chemical cause.

This status turn only ran the bundled-CPython collector
`outputs/hpc-submissions/collect-source128-guarded-restart-attempt001.py --tag news001`
and standard-library saved-result analysis. No source change, new job,
cancellation, model rerun or gate waiver occurred. Runtime and retention must
be addressed and validated before a new candidate can qualify; no fix or
restart is adopted by this status update.

## Full CPU gate passed; native/stability GPU queued, 20 October 2026 09:25 DTU-local

The [07:25:05 UTC snapshot](../outputs/hpc-results/source128-corrected-chemistry-guarded-restart-attempt001/status-cpu004.json)
confirms CPU **29760552 DONE**, after-CPU **29760553 DONE**, saved `cpu_gate`
passed with return code 0, and original repair profile passed with return
code 0. Unit evidence remains **1,252 passed, 66 CUDA-only skips, zero
failures/errors**, JUnit 137.853s. This establishes the full allocated CPU
stage, not native GPU or 2,500-update stability success.

Actual native/stability GPU **29760576** and callback **29760577** were
submitted at 09:23:22 DTU-local and are both **PEND**: one 80GB A100,
16 CPU slots, 24h allocation. The eligible resource wait is `Job slot limit
reached` across five hosts, with 12 CPU slots reserved; no ineligible-host
wait is reported. LSF's no-sooner-than **21 October 07:19:58 DTU-local** is
a changeable scheduling lower-bound estimate, not a promised start or finish.
**Training is not submitted or restarted.** Only a fully passing native and
2,500-update stability proof, followed by fresh identity/resume/quota checks,
may authorize the new continuation from restored epoch 36.

The `--tag cpu003` collector failed parsing `after-cpu-intent.json`, whose
contents are a plain SHA line rather than JSON. The
[collection-failure record](../outputs/hpc-results/source128-corrected-chemistry-guarded-restart-attempt001/status-cpu003-collection-failure.json)
is preserved. The external metadata collector now retains raw content and
`parse_error` and continues; `--tag cpu004` succeeded. These are bundled-
CPython invocations of the collector below, not a model/job/network failure,
an experimental rerun or a sealed-source change.

## Allocated unit tests passed; full CPU gate pending, 20 October 2026 09:22 DTU-local

The [07:22:19 UTC snapshot](../outputs/hpc-results/source128-corrected-chemistry-guarded-restart-attempt001/status-cpu002.json)
records CPU **29760552 RUN** and after-CPU **29760553 PEND**. Its saved JUnit
has **1,318 collected: 1,252 passed, 66 CUDA-only skips, zero failures/errors**,
duration **137.853s** (pytest stdout reports 138.08s), with three existing
`record_property`/JUnit-schema warnings. This is allocated CPU fixture proof,
not native GPU validation. The original CPU profile process is still running
and `cpu_gate` is absent: **the full CPU gate is not yet passed**, GPU
native/stability validation is not yet queued, and training is not queued.

The root collected saved scheduler/JSON/JUnit/log evidence using bundled
CPython and these command tails, without numerical model execution:

```text
outputs/hpc-submissions/collect-source128-guarded-restart-attempt001.py --tag submitted001
outputs/hpc-submissions/collect-source128-guarded-restart-attempt001.py --tag cpu001
outputs/hpc-submissions/collect-source128-guarded-restart-attempt001.py --tag cpu002
```

## Source128 sealed and CPU submitted, 20 October 2026

[Immutable preparation](../outputs/hpc-submissions/source128-corrected-chemistry-guarded-restart-attempt001-preparation.json)
binds 16 explicit overlays, including uncommitted implementation, tests and
configuration. [Staging/queue receipt](../outputs/hpc-submissions/source128-corrected-chemistry-guarded-restart-attempt001-stage-and-queue-001-output.json)
records return code 0, all **580 archive members authenticated**, and actual
CPU **29760552** plus after-CPU **29760553** submitted as
`submitted_not_passed`. GPU native/stability validation and training are
**not yet submitted**. Preparation's earlier `sealed_not_submitted` status is
historical; the separate queue receipt records the later submission.

| Identity | SHA256 |
| --- | --- |
| Source archive | `d7d6199482fb80f6447aca498fe657d3a397e67838e4edb8097fb1f7b26ffe07` |
| Source manifest | `06ce2313b1b0a2f2cfbd97e7322c3dad68c495d1a6e0ee926daa1c3152c37f35` |
| Immutable plan | `6baa24c563d6884f2a961339bd6f429d7d7bdd021d9b9128d0b86450bb9cf9d8` |

The root ran the following command tails via bundled CPython
`C:/Users/nikol/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`:

```text
outputs/hpc-submissions/prepare-source128-guarded-restart-attempt001.py
outputs/hpc-submissions/operate-source128-guarded-restart-attempt001.py stage-and-queue
```

These perform immutable packaging, transfer/staging authentication and
scheduler submission, with staging and queueing in one SSH operation. They
are not local numerical tests. Source128 requires fresh checks of its changed
science, without adopting old-objective native proof. The declared runtime
factor remains 1.25 with the same 24h allocation/23h internal caps. A source
seal and CPU submission do not establish CPU, native, stability or training
success; source127 remains stopped.

## Initial implementation and requirements, 20 October 2026

Initial status: **implementation prepared; review/validation pending**.
At that stage, no sealed new source release, job, validation pass or training
restart was established.
All numerical tests, model evaluation and update proofs run on allocated DTU
HPC. Local work is editing, static review, immutable packaging and saved-evidence
inspection only.

[Read-only preparation preflight](../outputs/hpc-results/source128-preparation-preflight001.json)
at 07:05:50 UTC/09:05:50 DTU-local succeeded: the stopped source127 jobs had
aged out of live scheduler records, `gpua100` was Open/Active with a 4,320-minute
queue maximum, and official work3 quota was 89.26/100 GiB used. This does not
prove a new job exists or reserve space. New allocations remain capped at 24h;
the queue's higher maximum does not amend that limit.

The user authorized correcting partial-loss scaling, adding early fixed-panel
checks and logging chemical versus representation failures separately. This
supersedes the stop only for a new immutable plan whose required checks pass.
**Never restart source127 or rewrite its stopped/failed evidence.** Its
[saved investigation](66_failure_guidance_execution_speed.md#saved-evidence-investigation-after-the-user-stop-20-october-2026)
does not establish the chemical cause of the deterioration. Normalization is
an evidence-backed design concern, not a proved sole cause.

## Immutable parent and unchanged scientific scope

Authenticate completed epoch 36, including model, AdamW, native RNG and separate
fresh-prior RNG. Its checkpoint SHA256 is
`3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f`.
Retain the frozen epoch-35 reference and original repair ramp; do not start a
new ramp or adopt the interrupted epoch-37 state. Preserve:

- 256 real and 512 fresh-prior samples per update (1:2), microbatch 128,
  12 search workers plus four main Torch threads within 16 CPU slots;
- 0.25/0.75 repair branch means, other loss weights, composition/occupancy/KL
  semantics and bond-only production search settings;
- joint encoder/decoder training, authenticated optimizer/RNG lineage,
  100 total epochs, serial continuation, at most 24h allocation and 23h internal
  runtime, including persisted deadlines and honest partial completion;
- every source/input, native-gradient, finite-value, parity, retention,
  sulfur/composition, memory, resume, runtime and fresh-quota gate.

The implementation explicitly selects `loss.version =
"repair-failure-guidance-v2"`. For a partial witness with `m` changed unordered
bonds, use the **sum** of their categorical NLLs divided by
`N + K + K*(K-1)/2`, the same decision-count denominator used by a complete
target (`N` is maximum slots, `K` is occupied slots). For `N=50, K=22`, the
denominator is 303 rather than `m=1..6`. Unedited partial labels contribute
zero; the existing all-attempt row mean, component weights and real/prior
branch weighting remain unchanged. Historical v1 retains its edited-label
mean and sealed records. This corrects relative label scaling without
establishing that normalization was the sole cause or that v2 is healthy.

Bind the explicit objective version and normalization to the new immutable
source/configuration. Do not claim mathematical equivalence to source127 or
reuse its old objective pass as validation of changed normalization. Bind every result
to the new source/plan, data, parent, reference and configuration. Any declared
runtime allowance must remain explicit and authenticated; no other gate is
waived by the restart authorization.

## Early checks and retained failure evidence

The declared `protocol.early_health` version is `chemistry-early-health-v1`:
first check after **32** updates from the epoch-36 handoff, then at every
multiple of **100** updates. This is 32, 100, 200, ...; it is not a cadence
restarted for each allocation. Each observation repeats the fixed-seed
1,024-prior panel and 256 held-out reconstruction examples. Preserve all
training RNG streams and every module's prior train/eval mode around these
observations. Authenticate the classified epoch-36 baseline separately.

Keep every existing health/retention check against both reference 35 and the
starting epoch-36 panels. In addition, raw-valid fraction may drop at most
0.02 from epoch 36, and **strict-teacher-valid fraction may drop at most 0.02**
from its newly measured classified epoch-36 baseline. Raw and strict baselines
are distinct; do not assume the old 641/1,024 raw-valid count is the strict
teacher count. Retain reconstruction, size, nonempty, composition, charge and
sulfur limits, rather than accepting one aggregate validity number.

For each fixed panel, save bounded original hard labels and the actual atom,
charge and symmetric-bond categorical marginals for at most **four examples
per failure kind** (chemical, representation, input). The nondifferentiable
classifier records graph sanitization, raw chemical validity, serialized-SMILES
parsing, schema encoding/canonical roundtrip and strict teacher acceptance as
separate stages. `None` means a later stage was not evaluated, never a pass.
Raw chemical validity retains the project's nonempty, sanitized, connected,
radical-free rule; sanitization alone is a narrower check. Unexpected
classification/encoder exceptions remain errors rather than hidden ordinary
invalidity. No classification step repairs a graph or contributes gradients.

On early-health failure, persist the observation and a diagnostic model/AdamW/
native-and-fresh-RNG snapshot with source/plan/parent/health identities. Mark it
**nonpromotable, epoch-incomplete and unusable for resume**; scheduler and
trainer checks reject it. Snapshot-capture failure is itself retained and
cannot permit continuation. These diagnostic states solve the previous loss
of failure evidence without becoming completed checkpoints.

## Staged proof before conditional continuation

1. Seal the implementation, tests, explicit early-check schedule, acceptance
   thresholds and update budget. Static review and packaging are preparation,
   not numerical validation. Run allocated CPU contracts and applicable native
   CUDA/parity/finite-gradient checks before the longer proof.
2. Authenticate and evaluate the epoch-36 baseline on the same fixed 1,024-prior
   seed/panel and 256 held-out reconstruction examples. Save graph/diagnostic
   evidence needed to distinguish rejection causes. Preserve evaluation RNG
   separately from training RNG so health checks do not change training draws.
3. Run **2,500 disposable matched updates** from epoch 36, covering the
   previously observed roughly **1,500–2,300-update deterioration window**;
   an eight-update readiness benchmark cannot demonstrate this. Restore the
   actual training shuffle, epoch-37 learning rate, AdamW and native/fresh RNG
   streams. The declared schedule contains 26 post-update observations:
   32 and 100, 200, ..., 2,500, in addition to the starting baseline. Retain
   baseline, intermediate, endpoint, failed/partial and timing evidence.
4. Apply early fixed-panel health gates during the proof and future training,
   rather than waiting for the 6,596-update epoch endpoint. Check raw validity,
   representation acceptance, reconstruction, composition/sulfur and existing
   retention limits using matched panels and declared thresholds. Failed
   scientific checks, exceptions, nonfinite values and resource exhaustion stop
   progression; finite training losses alone do not establish health.
5. Require the complete native/parity/coverage/retention/runtime/resume/quota
   gate set. Persist mean and conservative runtime estimates, actual resources,
   all gate outcomes and exact scheduler receipts. Include fixed-panel overhead
   in conservative epoch fitting; observation cost is not free. A test pass is not a queued
   continuation, and a submission is not a started or completed epoch.
6. **Discard the entire disposable test state**, including optimizer and RNG
   changes. Restore and authenticate the original completed epoch-36 state
   before a separate fresh continuation toward epoch 100. Check no competing
   training/callback remains active and recheck quota immediately before any
   submission. Only a fully passing, authenticated new plan may continue.

The implementation separates `chemistry_restart_policy.py` (metadata guard
shared by scheduler and trainer), `chemistry_guarded_validation.py` (allocated
GPU disposable proof), `chemistry_early_health.py` (policy, observations and
nonpromotable diagnostics), and `scripts/hpc/chemistry_guarded_restart.py`
(staged controller). CPU contracts include `test_chemistry_guarded_restart.py`
and the loss/classification/early-health suites; applicable native GPU checks
and the complete 2,500-update proof remain mandatory. File presence and static
review are not executed-test results. Packaging/staging commands are recorded
above; numerical commands and outcomes will be recorded from allocation evidence.

## Separate diagnostics and honest denominators

Report raw chemical validity from the native graph inspection independently
of schema encoding and canonical roundtrip acceptance. Preserve separate
nonempty/connectivity/native-check failures, unsupported representation,
encoding/roundtrip failures and unexpected errors, with attempts and applicable
denominators. A teacher-already-valid count includes extra requirements and
must not be renamed raw RDKit validity. Keep bounded-search accepted,
partial/censored, unsupported and error outcomes distinct from chemical and
representation checks; these describe different stages and can overlap.

Log the complete/extra-valid/partial/local-risk objective contributions and
their normalization. Per-term values are not whole-gradient ratios or causal
proof. Save enough per-row diagnostics or authenticated graph evidence to
investigate future deterioration; a hash chain alone cannot reconstruct lost
graphs. No sulfur, atom/bond mechanism or population improvement claim follows
from changing training batches without matched evaluation.

Source128 preparation and submission identities are recorded above. Allocated
test commands/results, later resources, outcomes and conditional restart
receipts will be appended to the experiment ledger when they exist. The
source127 stop and its missing epoch-37 checkpoint
remain historical facts regardless of the new plan's eventual outcome.
