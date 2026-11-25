# Sequential 500-update Boltz-only versus Boltz+DeFoG comparison

## Status and question

**Completed guided report available, 18 October 2026 21:26 DTU-local:**
[open the offline DeFoG-guided HTML](../outputs/hpc-results/source120a004-boltz-defog.html),
a verified byte-identical copy of the saved HPC report with 501 frames, states 0–500.
It preserves invalid graphs and predates fresh-pose evaluation; copying added
no model predictions, resampling, repair or rendering experiment.
[Manifest](../outputs/hpc-results/source120a004-boltz-defog.manifest.json)
and [verification](../outputs/hpc-results/source120a004-boltz-defog.verification.json).

**Guided arm completed; evaluation queued, 18 October 2026 21:19 DTU-local:**
source120 guided29748197 completed **500 accepted/500 attempted updates**,
without a stall, and authenticated saved native/terminal proof validation
passed. The stage took48,520.78s. This used sealed `armijo_v1`; future v2 was
not used. The control's healthy225-update stall remains unchanged. Guided's
raw endpoint has0/9 valid candidates and no argmax SMILES; no fresh binding
gain is established. The authorized callback submitted evaluation29750141 /
after-evaluation29750142, both **PEND**. These queued followups are not a
completed evaluation or matched full-budget comparison.
[Saved status proof](../outputs/hpc-results/status-20261018-evening.json)
and [ledger](05_experiment_ledger.md). No job/settings changed in this status read.

**Future-policy boundary, 18 October 2026:** the user subsequently authorized
`best_trial_v2` for future latent-optimization runs. That policy applies a
nonincreasing full step immediately; otherwise it takes the first strictly
improving halved step or applies the lowest-loss finite
trial when none improves, so loss can rise while the requested budget
continues. Final source122 attempt003 CPU job 29749480 passed 423 engineering
fixtures across 16 modules, with no failures/errors/skips, including the
full-step-equality amendment. This is not native v2 Boltz validation. The first
failed transfer and attempt002's intermediate 411-pass result are preserved.
[Verified final CPU evidence](../outputs/hpc-results/source122-best-trial-cpu-002/local-verification.json)
and [docs62](62_full_budget_best_trial_optimization.md) record the scope and
[failed final transfer](../outputs/hpc-submissions/source122-best-trial-attempt003-upload-failure.json).
**This source120 comparison
retains its sealed `armijo_v1` protocol.** The recorded 225-update healthy
control stall, active guided protocol, native acceptance/terminal checks and
full-budget flags below are unchanged. No historical run is rerun or relabeled
by the future policy authorization.

**Control stalled with accepted proof; guided running, 18 October 2026 11:36
DTU-local:** source120 attempt004 control29747656 DONE after225 accepted /
226 attempted updates and a verified Armijo stall, using about3.2h of optimizer
execution. Native derivative, prefix, accepted-update descent, unchanged-point
and terminal checks passed. The saved result marks healthy terminal acceptance
true and full500 completion false. This is not convergence; control
checkpoints250/500 remain `not_reached`.

Callback29747657 independently authenticated scheduler completion and the saved
native proof, then submitted **guided29748197 / callback29748198** at04:26
DTU-local. Guided29748197 is **RUN**, with280 accepted updates in the saved
snapshot (last metrics row279). That snapshot is progress, not guided terminal
acceptance or a completed comparison. No evaluation allocation is queued and
no fresh-pose followup result exists yet.

| Saved record | DBMol binding-only signed/scaled loss |
| --- | ---: |
| Shared initial record | 7.836312 |
| Control terminal,225 accepted | 5.879790 |
| Guided last metrics row279,280 accepted | 5.190700 |

These are the fixed-context binding component, excluding repair loss. At the
guided row279, repair guidance/target/gradient capture are active, coefficient
0.1, raw repair gradient norm91.3647, weighted norm9.13647 and binding gradient
norm36.36769. Its weighted repair loss is42.02894 and total loss47.21964;
that total is not the binding-only quantity in the table. The arms currently
have different accepted budgets and the guided arm is unfinished, so these
records do not establish a final advantage or fresh native binding gain.
Each arm's latest saved hard panel has0/9 valid and `argmax_smiles=null`;
this statement describes those latest panels, not all historical samples.
The [offline control HTML report](../outputs/hpc-results/source120a004-boltz-only-control.html)
is a byte-identical copy of the saved HPC artifact, with
[verification](../outputs/hpc-results/source120a004-boltz-only-control.verification.json).
It was not rerendered, resampled or repaired for this status read; reporting
does not validate the molecular outcomes.

[Verified capsule003](../outputs/hpc-results/source120a004-staged-003/local-verification.json),
[control proof validation](../outputs/hpc-results/source120a004-staged-003/submissions/source120a004-boltz-defog-staged-24h-attempt004/stage-validation/control/result.json),
[successor submission](../outputs/hpc-results/source120a004-staged-003/submissions/source120a004-boltz-defog-staged-24h-attempt004/after-control.json)
and [ledger](05_experiment_ledger.md). No manual submission, cancellation,
setting change or gate change occurred in this status turn; the guided jobs
were submitted by the previously authorized authenticated callback.

**Latest queue status, 18 October 2026 00:39 DTU-local:** source120 attempt004
Boltz-only **control29747656 PEND / callback29747657 PEND**. LSF reports a
job-slot limit at the eligible GPU host; four slots are reserved on another
host. No arm has started, no new trajectory exists and no guided/evaluation
successor has been submitted. CPU29747652's341 checks plus actual source109
preparation/validation remain historical passes. This resource wait does not
establish invalid arguments or a new gradient/model failure. No job, setting
or check changed in this read-only status turn.
[Verified capsule002](../outputs/hpc-results/source120a004-staged-002/local-verification.json)
and [ledger](05_experiment_ledger.md).

**Staged replacement queued, 17 October 2026 22:55 DTU-local:** source120
attempt004 CPU29747652 DONE,341 allocated checks passed plus successful actual
historical preparation/validation. Control **29747656 PEND** requests18h;
callback **29747657 PEND**. Guided24h and evaluation6h stages submit only after
the authenticated predecessor and unchanged scientific checks pass. No arm
has started. The old source11948h request was intentionally retired while PEND,
confirmed EXIT; all earlier evidence remains. Failed collection/path fixtures
in source120 attempts001/003 and held-unsubmitted002 are retained in the ledger.
[Verified capsule](../outputs/hpc-results/source120a004-staged-001/local-verification.json),
[control receipt](../outputs/hpc-results/source120a004-staged-001/submissions/source120a004-boltz-defog-staged-24h-attempt004/stage-submissions/control-receipt.json)
and [ledger](05_experiment_ledger.md).

This source preserves all477 source119 scientific members and adds only the
stage runner/controller and their tests. Archive
`81788ed64318b743f8d8f31eefcc264048ce33b6face715cf3bdb93fc69ade4e`,
manifest `026668cba0cea315b529bd3be459ce70e88aae30007ed0ee8ef3ba7237a8da37`;
[preparation](../outputs/hpc-submissions/source120a004-preparation.json) and
[campaign](../outputs/hpc-results/source120a004-staged-001/submissions/source120a004-boltz-defog-staged-24h-attempt004/campaign.json)
pin the source, plan, readiness and unique output roots. All old native gates,
500-update/stall semantics and frozen epoch89 decoder remain in force.

**Historical planning snapshot, 17 October 2026 22:39 DTU-local, superseded by
the actual status above:** the user now
prefers allocations at most24h. Replace the earlier single48h request with
serial control18h/17h internal, guided24h/23.5h internal, then
evaluation/report6h/5.5h internal. The optimization caps remain15h and23h,
and followups remain2h per arm; total allocated48h/internal46h is unchanged.
Split only between completed independent arms/stages, with authenticated
terminal and scientific checks before each successor. This is not a mid-arm
optimizer resume or permission to label a partial result full500. All frozen
inputs, scientific objectives, repair schedule, native/descent/stall semantics
and evaluation memberships remain declared as before. At22:39 source119 was
still PEND; no retirement or replacement receipt exists yet. The following
48h submission/audit records are historical, not altered sealed jobs.

**Request audit, 17 October 2026 22:30 DTU-local:** source11929747450 remains
PEND. The request is48h for both serial arms together, with46h internal:
15h control,23h guided,2h followups per arm and4h startup/report reserve.
The actual four-slot/64GB/one80GB-GPU request is compatible with inspected
host inventory and the live72h queue maximum; availability still prevents
allocation. GPU `aff=no` and requested MPS/exclusive-process options are valid
declared controls, not proof that resources are free. The public DTU GPU page
still states24h, unlike the live queue's72h setting. No job or argument changed.
[Raw scheduler/host audit](../outputs/hpc-results/lsf-argument-audit-20261017/independent-readonly/scheduler-host-audit.json)
and the [ledger](05_experiment_ledger.md) retain exact options and official
documentation links. No start ETA is guaranteed.

**Latest queue status, 17 October 2026 22:14 DTU-local:** source119
**GPU29747450 PEND / not started**, awaiting available GPU resources; CPU29747447
readiness remains DONE. No optimization campaign output exists yet. The
comparison uses the original frozen epoch-89/two-round non-triangle VAE,
identically in both arms; see [exact decoder identity](#frozen-scientific-identity).
The separate source118 triangle-VAE chemistry benchmark/promotion remain
pending. No new submissions, settings or model runs occurred in this status
turn, and no start ETA is established.
[Verified capsule004](../outputs/hpc-results/source119-recovery-004/local-verification.json)
and [scheduler snapshot](../outputs/hpc-results/source119-recovery-004/scheduler_snapshot.json).

**Earlier recovery passed; comparison queued, 17 October 2026 21:36
DTU-local:** source119 CPU29747447 DONE. All267 allocated checks passed
(247 original plus20 regressions; no skips), followed by successful preparation
and validation against the actual authenticated historical pilot. This
establishes the bookkeeping recovery on its real inputs, not new500-step native
behavior. **GPU29747450 is PEND / not started** for the same48h sequential
comparison. The [verified GPU receipt](../outputs/hpc-results/source119-recovery-003/submissions/source119-boltz-defog-500-attempt001/comparison-submission-receipt.json)
binds the passed CPU readiness result; [capsule003](../outputs/hpc-results/source119-recovery-003/local-verification.json)
and [scheduler snapshot](../outputs/hpc-results/source119-recovery-003/scheduler_snapshot.json)
confirm the new job. Native checks, optimization and newly generated reports
remain future work in that allocation. No native long-run result is claimed.
[CPU completion](../outputs/hpc-results/source119-recovery-002/submissions/source119-boltz-defog-500-attempt001/readiness/completion.json)
records the checks. The complete chronology remains in the [ledger](05_experiment_ledger.md).

**Earlier recovery submission, 17 October 2026 21:34 DTU-local:**
source119 CPU29747447 is RUN (four slots/16GiB total/one hour). At the
[verified capture](../outputs/hpc-results/source119-recovery-001/local-verification.json),
no readiness stage had completed. **The GPU comparison is not yet submitted**;
allocated tests plus actual historical-receipt preparation/validation must pass
first. Source119 preserves475/477 source114 scientific members byte-for-byte;
only the runner and regression tests change. The adapter leaves final controller
acceptance untouched and derives optimization completion from authenticated
25-update terminal receipts before applying the unchanged scientific checks.
No scientific setting,500-update budget or initial25-update repair ramp changes.
The separate VAE suite is untouched.

[Source119 preparation](../outputs/hpc-submissions/source119-preparation.json)
and [actual CPU receipt](../outputs/hpc-results/source119-recovery-001/submissions/source119-boltz-defog-500-attempt001/readiness-submission-receipt.json)
pin archive `119150fa8e9e70b5e22bdedef0f00f53bafb896b938965cb9d04f9c0214e932c`,
manifest `444d1947aac681c0e8f9b3976d1a13d3cba70dfb1a2f9336bf5f11d60720a286`
and bootstrap `3ce32a049c4001f2fa73a8c2404775059792eb528c3ca38ee9d74352bf4b6e9f`.
The full chronology and unique recovery paths are in the [ledger](05_experiment_ledger.md).
This is a bookkeeping recovery prerequisite, not a passed native long run.

**Latest result, 17 October 2026 21:05 DTU-local:** source114 **29741066 EXIT1**
after running20:52:08–20:52:54. Its247 allocated preflight checks passed in24.19s;
preparation failed7.19s later while authenticating historical pilot receipts.
No new optimization arm started. The final source109 controller case has
`status="passed"`, whereas the reused optimization-completion validator requires
`status="completed"`. The original25-update completion, frozen-state and guard
evidence all remain passed. This is an established launcher bookkeeping error,
not a new Boltz/DeFoG gradient failure or model incompatibility. Quota passed
(10.06GiB free versus4GiB required); neither OOM nor walltime caused the exit.

[Verified failure evidence](../outputs/hpc-results/source114-failure-001/verification.json)
and [saved JSON/AST analysis](../outputs/hpc-results/source114-failure-001/failure-analysis.json)
retain the original logs and receipt identities. Archive SHA256:
`e543040fafd8e4c58866d24647dccf806a56ee272fe18ef5e262f1fd4e552397`.
A future fix must obtain optimization completion from the authenticated terminal
trajectory/arm summary without rewriting the final controller status or waiving
checks. No code changes, resubmission, new scientific tests or model calls were
made during this status investigation. See the [ledger](05_experiment_ledger.md).

**Historical queue observation, 17 October 2026 12:42 DTU-local:** `gpua100` is
`Open:Active`; one listed 80 GB host (`n-62-18-10`) is `Busy`, the other four
remain `Down`. Source114 **29741066 remains PEND**, now with “Job slot limit
reached: 1 host.” This is partial recovery; the long comparison has not started
and no start/recovery ETA is confirmed. No jobs/settings or numerical tests
changed. [Structured receipt](../outputs/hpc-results/a100-status-20261017-124221.json).

**Earlier queue observation, 17 October 2026 00:58 DTU-local:** `gpua100` is
`Open:Active` again, but **29741066 remains PEND** with “There are no suitable
hosts for the job.” All five eligible 80 GB hosts are listed `Down`; the checked
host `n-62-18-8` is `closed_Adm`. This is not evidence of a hardware fault.
The timing is consistent with the announced service window, but neither an
A100-specific cause nor a recovery ETA is confirmed. Queue reactivation is not
host availability or an optimization start. No jobs/settings changed.
[Structured receipt](../outputs/hpc-results/a100-queue-reactivated-20261017.json)
and [ledger](05_experiment_ledger.md) retain observed/inferred/unknown scope.

**Earlier restored-access observation, 17 October 2026:** source114 **29741066
PEND**, with scheduler reason “The queue is inactivated by the administrator”;
`gpua100` reports `Open:Inact`. The comparison has not started. The
[restored-access scheduler receipt](../outputs/hpc-results/status-restored-20261017.json)
records the live read. No resubmission, cancellation, model run or settings
change occurred in this status turn. The separate VAE chemistry benchmark's
retrieved result is now explained by a runtime-fit gate, not an unresolved
comparator failure; its callback stopped and no continuation began. See the
[ledger](05_experiment_ledger.md).

**Earlier access-limited observation, 17 October 2026:** the loaded DTU key was accepted
but SSH then required additional password authentication. No fresh scheduler,
preflight or optimization result was retrieved. The PEND record below is the
last verified 16 October snapshot; it does not show whether the job started
overnight. No job or VPN/SSH setting was changed. See the
[access receipt](../outputs/hpc-results/status-access-20261017.json) and
[ledger](05_experiment_ledger.md).

**Submitted, 16 October 2026 16:25:58 DTU-local:** source114 job **29741066**
was queued in `gpua100` at the downloaded observation for one sequential comparison, at most 500 updates per
arm. The downloaded [submission receipt](../outputs/hpc-results/source114-submission-001/submission-receipt-002.json)
and [scheduler observation](../outputs/hpc-results/source114-submission-001/scheduler-status.json)
confirm **PEND**, waiting for sufficient slots/resources for the whole requested
duration on five hosts; eligible pending time was 35 seconds and ineligible time
zero. No manual hold was reported. **At that observation allocated preflight
and native checks had not run; no new optimization result was established.**

The immutable [preparation record](../outputs/hpc-submissions/source114-preparation.json)
retains its historical `packaged_not_submitted` status; the later durable receipt,
rather than rewriting preparation, establishes the actual submission.
Full chronology is in the [experiment ledger](05_experiment_ledger.md).

The [submission verification](../outputs/hpc-results/source114-submission-001/verification.json)
passed after download: 477 scientific member hashes, manifest/archive and
bootstrap identities, identical original/recovery `bsub` commands and actual
job receipt/PEND status. Command:

```powershell
C:/Users/nikol/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe outputs/hpc-submissions/verify-source114-submission-001.py
```

This is standard-library byte/JSON verification only, not a scientific test.
Model code was frozen at sealing and has not been edited afterward.

| Identity | SHA256 |
| --- | --- |
| Source114 archive | `9299a6617efaf8745c36e7d3cb42ad74d3d26ddc75b6870b90f9eeef0bc293bb` |
| Source114 manifest | `af40fa2cd094e72b4f36b3083883bc83112d1784b778e1df155f057e0e7db8ca` |
| Bootstrap | `cfabb655d1bc3e5ac1a14e2ca9ead335cf23006de603b7f7f7a3b765919fbc91` |
| Completed source109 reference pilot plan | `edc530c5357d86ea2f5992788a86c4b9ff4302150c959d79e4f4f8db434eb295` |
| Completed source109 pilot result | `4edfb59cc577c98140445360af9c854cdddaca9c833168233cf2f9ee47d6725a` |

Staging authenticated 538 members. The release preserves 456 inherited source109
files unchanged and records 21 file deltas, including operational limits, long
runner/evaluation, report capture/assets and focused tests. The full preparation
record lists every changed-file hash. The exact runtime plan and arm configs
will be generated/authenticated inside the allocation and must be retained as
new artifacts; their hashes are not invented before that stage.

The first submission operator failed **before spawning `bsub`** with
`FileNotFoundError('bsub')` in a non-login SSH environment. Its exclusive
prelaunch intent and absence of receipt are preserved. Recovery operator002
used DTU login `bash -lc` with the same immutable command/source and a separate
receipt; only job 29741066 was actually submitted. An earlier SSH session reset
was followed by successful bounded SSH/SFTP operations; no VPN or SSH settings
were changed. This was an operator-path failure, not a native/model test failure.

The completed source109 pilot established that both branches can execute 25
accepted updates under the declared pointwise-native derivative contract. It
did not produce valid raw endpoint molecules or a demonstrated fresh binding
gain. The longer comparison asks whether a further optimization budget changes
hard-molecule validity, composition, repair closeness or binding predictions.
It is one paired trajectory, not a replicated test of a general chemical
advantage. The historical 25-update results and failed numerical diagnostics
remain unchanged.

## Frozen scientific identity

Both arms start from the same authenticated saved latent vector and protein
context. Keep the original frozen validated VAE decoder, Boltz-2 affinity and
confidence checkpoints, target protein/pocket, native preparation, temperature,
graph vocabulary and supported charge policy. The fixed initial occupancy mask
contains **22 slots**. Only molecular `z` changes; no encoder, decoder, Boltz or
DeFoG weights are trained. In particular, this comparison does not select an
unvalidated chemistry-continuation decoder or use the unstarted 1:3 VAE branch.

The exact decoder is the **original epoch-89 source047 stabilized VAE**,
`stable_deep_pair_refine`, trained on215,000 real molecules. The
[actual source119 YAML](../outputs/hpc-results/source119-recovery-003/submissions/source119-boltz-defog-500-attempt001/readiness/prepared/boltz_only.yaml)
pins `/work3/s234818/latentmol/runs/20261001T001617Z_vae-train_c008c090/best_validity.pt`,
SHA256 `1893ddc183473fd4f27c6cd921d8b302f67ab388b660ada44bf3a2d6d5e28088`.
Its latent width is256, atom hidden width96, pair width48 and it has **two
refinement rounds without triangle updates**, with the historical global trunk.
The preserved [training record](../outputs/hpc-results/source078-dense100/extracted-single003/original/training-run.json)
and [architecture comparison](52_large_corpus_triangle_vae.md) establish that
identity. Both optimization arms use this same checkpoint and initial `z`;
DeFoG guidance changes the loss applied to `z`, not decoder weights.

This is distinct from the **new four-round triangle VAE**, jointly trained on
1,688,539 real molecules through completed epoch35. That newer parent remains
preserved for the separate source118 chemistry-continuation campaign; its
1:3 fresh-prior continuation has not yet started. It is also distinct from
historical decoder-only chemical fine-tuning checkpoints. No decoder was
silently switched between these studies.

The control minimizes the existing DBMol binding objective. The guided arm
uses the same current relaxed shared molecule in that binding branch and adds
the existing fixed-target DeFoG categorical NLL. Sampling, DeFoG's discrete
trajectory and target selection remain outside autograd. A repaired graph is a
separate stopped target, not a replacement identity fed to Boltz. Existing
compatibility, validity, count and slot-alignment checks still apply to targets;
failed attempts stay in the record and follow the declared fallback.

The intended changes from the validated short comparison are the requested
update budget, checkpoint coverage and associated execution/output budgets.
Any additional science or validation difference must be explicit in the sealed
configuration and ledger, rather than hidden in a script default.

The absolute schedule implementation is
[`repair_guidance/config.py`](../src/latentmol/repair_guidance/config.py),
`linear_updates_v1`. It reproduces the original fractional arithmetic over
indices 12–24 before holding the endpoint, avoiding a new rounding rule.
The shared accepted-step reader takes an explicit `requested_updates=500` for
this comparison; its default remains 25 for historical pilot verification.

## Keep the initial repair ramp

Run **at most 500 requested updates per arm**, sequentially within one GPU allocation.
The control has zero repair weight and no active ramp/provider. The guided arm
retains source109's initial 25-update schedule and then holds its final weight:

| Zero-based update index | Guided repair coefficient |
| --- | --- |
| 0–12 | 0.001 |
| 13–24 | Linear increase from the original 25-update schedule, reaching 0.1 at index 24 |
| 25–499 and endpoint | 0.1 |

This preserves the old rule `progress = min(index, 24) / 24`, start fraction
0.5 and end fraction 1.0. Increasing the requested budget must **not** stretch
that ramp across 500 updates. A stalled run does not compress or redraw the
schedule. Repair refresh, attempt/denoising budgets, seed ordering, selection
and normalization remain explicit configuration fields.

## Derivative and actual-update acceptance

Retain `pointwise_native_ad_v4`, the declared contact-head/assembly arithmetic,
native baseline and frozen-state parity, bridge audits, independent AD and
objective recomposition. Historical whole-network secant mismatches remain
observations; the local-AD certificate does not relabel them as passing finite
differences or claim smoothness across finite steps. See
[docs57](57_pointwise_gradient_validation.md).

Every accepted update must separately pass the existing native Armijo descent
and trust-region checks at its own context, fixed repaired target and repair
coefficient. The requested 500 updates do not authorize overriding a failed
gate or continuing a scientifically failed arm. Native audits retain the
five-update cadence, including the actual terminal state; the final plan must
authenticate this policy. A queued or running job alone is not evidence that
any of these checks passed.

A **verified Armijo stall** may terminate an arm before update 500 and allow
the other arm to run independently. This requires the entire declared trial
budget to be exhausted, the latent point to remain unchanged, and all prefix
native checks, accepted-update descent receipts and terminal receipts to pass.
It is an audited early terminal state, not an optimizer convergence claim or a
passed 500-update certificate. Real failed native gates, exceptions and resource
exhaustion stop the job; they are not classified as healthy stalls. Persist the
terminal kind, attempted/accepted update counts and separate full-budget flag.

## Checkpoint evaluation and presentation

Save comparison checkpoints at states **0, 25, 50, 100, 250 and 500**, retaining
the actual available states when a trajectory terminates early. Report planned,
attempted and accepted updates separately. At each available checkpoint retain
the relaxed shared graph and its raw decoder occupancy, argmax labels and
declared hard samples, validity/rejection reasons, composition/atom counts,
binding components, scheduled repair coefficient and target identity.
Raw invalid or disconnected outputs must remain unchanged.

The planned hard-candidate budget remains eight categorical samples plus
argmax per recorded state (nine records); the guided repair refresh remains every five updates with
four prescribed attempts. Checkpoints beyond an audited early terminal state
must be explicit `not_reached` memberships, never replaced by the endpoint or
quietly removed. Completed requested updates, engineering-prefix acceptance
and full checkpoint-evaluation coverage are separate result fields.

Use the same prescribed hard-sampling and bounded repair-audit budgets in
both arms. A found repair witness is an upper-bound witness under that search;
an exhausted search is censored, not proof that no nearby valid molecule exists.
Any fresh native binding evaluation requires a compatible valid hard molecule
and an actual new prediction; invalid/unavailable members remain explicit.
Fixed-pose proxy improvement does not establish fresh binding gain.

Fresh-pose membership is predeclared as six checkpoints × raw argmax or bounded
repair witness × pose seeds 17 and 23: **24 memberships per arm**. Selection is
by those rules, not native binding score. Here the `repaired` role is the
bounded rule-search witness from the evaluation audit, **not the native DeFoG
target** used during optimization. Prediction priority is fixed before outcomes:
seed 17 before seed 23, checkpoints 0/500/25/50/100/250 within each seed, and
raw before repaired. Memberships remain listed chronologically in reports.
The cap is at most 24 prediction
attempts per arm, with 1,800 seconds reserved for an eligible prediction;
the shared 7,200-second followup cap can leave eligible members budget-excluded.
Identical SMILES/pose-seed results may be reused with that fact recorded.
`invalid_raw`, `no_witness`, `not_reached` and `budget_excluded` are distinct
outcomes retained in the denominator, not completed predictions.
Budget-excluded or failed/unresolved evaluation memberships make the separate
evaluation-coverage result incomplete; they do not silently satisfy the
followup-completion gate. Invalid/no-witness/unreached outcomes can be fully
recorded without a native prediction and retain that reason.

Generate the standard interactive report from **saved results only**, with
overlaid losses, run/initialization selection, step playback, molecule snapshots
and the probability channels actually saved. Display target/coefficient/context
changes, incomplete runs and unavailable channels. Do not interpolate missing
states, regenerate molecules or hide invalid graphs for presentation. Keep raw
JSON/JSONL and manifests authoritative; renderer failures are separate from
scientific outcomes. See [docs59](59_interactive_optimization_reports.md).

## Execution and provenance

The **current source120 attempt004** uses three serial GPU allocations and
allocated CPU successor callbacks. Control18h/17h internal and guided24h/23.5h
internal start independently from the original saved `z`; evaluation/report
6h/5.5h internal follows only after both accepted terminal receipts. The
optimization caps15h/23h, followups2h per arm and total48h/46h reservation are
unchanged. Each GPU stage requests four slots/64GB host memory and one80GB A100;
callbacks use four CPU slots/16GB and30 minutes. No callback is a native
optimization stage.

The current entry points are
[`run_boltz_defog_long_stage.py`](../scripts/run_boltz_defog_long_stage.py) and
[`boltz_defog_staged_campaign.py`](../scripts/hpc/boltz_defog_staged_campaign.py).
They reuse the existing long plan and scientific readers. Every transition
requires actual scheduler DONE (or unique completed wrapper evidence after
genuine scheduler aging), unchanged source/configuration/frozen inputs,
hash-sealed stage artifacts, full native terminal/descent evidence and a fresh
official4GiB quota check. Revalidate failed/partial receipts as failures; do
not restart an arm, replace its initialization or infer science from DONE alone.
Current unique roots are in
[source120 attempt004 preparation](../outputs/hpc-submissions/source120a004-preparation.json).

### Historical single-allocation implementation

The source114/source119 plan below used one serial GPU job rather than parallel optimization arms or a new broad
diagnostic suite. Include the relevant CPU/static checks in that allocation
where feasible, then authenticate the source, plan/configuration and all model,
initial-state/context and historical readiness bindings before optimization.
No unallocated numerical work is authorized by this documentation task.

Its preserved implementation entry points are
[`scripts/run_boltz_defog_long_comparison.py`](../scripts/run_boltz_defog_long_comparison.py),
[`scripts/evaluate_boltz_defog_long_checkpoints.py`](../scripts/evaluate_boltz_defog_long_checkpoints.py)
and [`scripts/hpc/boltz_defog_long_job.py`](../scripts/hpc/boltz_defog_long_job.py).
The wrapper claims the allocated attempt exclusively, checks official user
quota, runs focused pytest preflight, authenticates/prepares and validates the
plan, rechecks quota and then runs the serial pair in that same allocation.
The minimum official free user quota is **4 GiB**; global filesystem free space
does not satisfy this guard. A duplicate allocation cannot replace the original
attempt record. Implementation/static review is not a passed allocated check.

Sealed source identity is `source-114-boltz-defog-500-attempt001`. A read-only connection
on 16 October confirmed that `gpua100` permits up to 72 hours. The submitted job
requests **48 hours**, one A100 80 GB physical GPU with
`mode=exclusive_process:mps=yes:j_exclusive=yes`,
and **four CPU slots × 16 GB host-memory reservation** (64 GB total), matching
the validated pilot's CPU/memory shape. The internal allocation budget is
**46 hours**:

| Budget scope | Seconds | Hours |
| --- | ---: | ---: |
| Boltz-only optimization | 54,000 | 15 |
| Boltz+DeFoG optimization | 82,800 | 23 |
| Control followups | 7,200 | 2 |
| Guided followups | 7,200 | 2 |
| Preflight/report reserve | 14,400 | 4 |
| Internal total | 165,600 | 46 |

These are enforced limits and reservations, not an ETA or guarantee that all
500 updates fit. Each arm additionally caps **20,000 tracked Boltz
forward/native-forward calls**, including native AD probes. One such call can
execute two physical Boltz trunks; internal activation-checkpoint recomputation
is not counted as a separate call. Therefore 20,000 is not a ceiling on every
low-level operation. The guard in
[`execution_budget.py`](../src/latentmol/execution_budget.py) checks time before
tracked calls and persists rejected attempts. Outer process deadlines also
bound setup, discrete denoising and a single native call that has not returned.
Budget exhaustion remains a failed/partial result, never a passed optimization.

The live quota read reported 89.51/100 GiB used (about 10.49 GiB free); no
capacity pass for a future submission is implied. Final resources, quota,
source/configuration identities and allocated quota must be rechecked by the
queued job before running.
That 16 October preparation query found the separate source112 chemistry
benchmark/callback in EXIT; their cause was unreviewed at the time, and scheduler
state alone did not establish a cause or chemistry start. The 17 October update
above now records the retrieved runtime-fit failure and stopped callback; the
earlier observation remains part of the history.

Actual remote locations:

- Release: `/work3/s234818/latentmol/releases/source-114-boltz-defog-500-attempt001`.
- Control/receipts: `/work3/s234818/latentmol/submissions/source114-boltz-defog-500-attempt001`.
- Campaign: `/work3/s234818/latentmol/runs/source114-boltz-defog-500-attempt001`.
- Scheduler logs: `/work3/s234818/latentmol/logs/source114-boltz-defog-500.29741066.{out,err}`.
- Saved-only reports after execution or partial failure:
  `<campaign>/reports/{boltz_only,boltz_defog,combined}/index.html` where artifacts
  are available. These report paths are planned outputs, not existing results.

The durable run record must include:

- Immutable archive/manifest, plan and arm-configuration hashes; exact source
  deltas from the validated release, including report capture assets.
- Initial `z`, decoder/scorer/DeFoG checkpoints, native context, pocket and
  occupancy-mask identities; RNG lineage and declared sampling seeds.
- Actual scheduler submission command/receipt, job ID, resources, time cap,
  host/environment and serial arm order, with persisted completion/failure.
- Actual check/audit/descent receipts, refresh attempts/acceptances, targets,
  coefficient schedule, attempted/accepted updates and checkpoint artifacts.
- Every configured evaluation membership, including invalid/censored/failed
  members, and report source/output hashes.

Do not overwrite earlier source109 runs or the independent source112/source113
campaigns. A resource/time-limit failure remains an execution result, not an
implicit request to loosen checks or change science. Any later continuation
requires an explicit authenticated resume contract and a new receipt; this
single-job plan does not claim automatic resumability.
